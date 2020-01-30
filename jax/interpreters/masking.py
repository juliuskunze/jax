from collections import namedtuple
from functools import partial
from itertools import chain

import numpy as onp
from contextlib import contextmanager
from .. import core
from ..core import Trace, Tracer
from ..util import safe_map, safe_zip, unzip2
from ..abstract_arrays import ShapedArray, Poly, eval_polymorphic_shape, \
  eval_polymorphic_dim
from .. import linear_util as lu
from . import polymorphic

map = safe_map
zip = safe_zip

shape_parameterized_primitive_rules = {}
masking_rules = {}

def defvectorized(prim):
  masking_rules[prim] = partial(vectorized_masking_rule, prim)

def defnaryop(prim):
  masking_rules[prim] = partial(naryop_masking_rule, prim)

def vectorized_masking_rule(prim, padded_vals, logical_shapes, **params):
  del logical_shapes  # Unused.
  padded_val, = padded_vals
  return prim.bind(padded_val, **params)

def naryop_masking_rule(prim, padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  return prim.bind(*padded_vals)

ShapeEnvs = namedtuple("ShapeEnvs", ["logical", "padded"])
shape_envs = ShapeEnvs({}, {})  # TODO(mattjj): make this a stack for efficiency

def is_tracing():
  return shape_envs.padded

@contextmanager
def extend_shape_envs(logical_env, padded_env):
  global shape_envs
  new_logical = dict(chain(shape_envs.logical.items(), logical_env.items()))
  new_padded = dict(chain(shape_envs.padded.items(), padded_env.items()))
  shape_envs, prev = ShapeEnvs(new_logical, new_padded), shape_envs
  try:
    yield
  finally:
    shape_envs = prev

def shape_as_value(shape):
  return eval_polymorphic_shape(shape, shape_envs.logical)

def shape_dim_as_value(dim):
  return eval_polymorphic_dim(dim, shape_envs.logical)

def try_get_shape_dim_as_value(dim, default=None):
  return shape_dim_as_value(dim) if shape_envs.logical else default

def padded_shape_as_value(shape):
  return eval_polymorphic_shape(shape, shape_envs.padded)

def padded_shape_dim_as_value(shape):
  return eval_polymorphic_dim(shape, shape_envs.padded)

def mask_fun(fun, logical_env, padded_env, in_vals, polymorphic_shapes):
  with core.new_master(MaskTrace) as master:
    fun, out_shapes = mask_subtrace(fun, master, polymorphic_shapes)
    with extend_shape_envs(logical_env, padded_env):
      out_vals = fun.call_wrapped(*in_vals)
    del master
  return out_vals, out_shapes()

@lu.transformation_with_aux
def mask_subtrace(master, polymorphic_shapes, *in_vals):
  trace = MaskTrace(master, core.cur_sublevel())
  in_tracers = [MaskTracer(trace, x, s).full_lower()
                for x, s in zip(in_vals, polymorphic_shapes)]
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_shapes = unzip2((t.val, t.polymorphic_shape) for t in out_tracers)
  yield out_vals, out_shapes

class MaskTracer(Tracer):
  __slots__ = ["val", "polymorphic_shape"]

  def __init__(self, trace, val, polymorphic_shape):
    self.trace = trace
    self.val = val
    # TODO this breaks tests but is used in aval:
    # assert self.val.dtype
    self.polymorphic_shape = polymorphic_shape

  @property
  def aval(self):
    return ShapedArray(self.polymorphic_shape, self.val.dtype)

  def is_pure(self):
    return all(type(poly) is not Poly or poly.is_constant
               for poly in self.polymorphic_shape)

  def full_lower(self):
    if self.is_pure():
      return core.full_lower(self.val)
    else:
      return self


class MaskTrace(Trace):
  def pure(self, val):
    return MaskTracer(self, val, onp.shape(val))

  def lift(self, val):
    return MaskTracer(self, val, onp.shape(val))

  def sublift(self, val):
    return MaskTracer(self, val.val, val.polymorphic_shape)

  def process_primitive(self, primitive, tracers, params):
    vals, polymorphic_shapes = unzip2((t.val, t.polymorphic_shape) for t in tracers)
    if primitive in shape_parameterized_primitive_rules:
      rule = shape_parameterized_primitive_rules[primitive]
      out, out_shape = rule(shape_envs, vals, polymorphic_shapes, **params)
    else:
      avals = [t.aval for t in tracers]
      out = primitive.abstract_eval(*avals, **params)
      out_shape = [o.shape for o in out] if primitive.multiple_results else out.shape
      logical_shapes = map(partial(eval_polymorphic_shape, values_dict=shape_envs.logical), polymorphic_shapes)
      masking_rule = masking_rules.get(primitive)
      if masking_rule is None:
        raise NotImplementedError('Masking rule for {} not implemented yet.'.format(primitive))
      out = masking_rule(vals, logical_shapes, **params)
    if not primitive.multiple_results:
      return MaskTracer(self, out, out_shape)
    else:
      return map(partial(MaskTracer, self), out, out_shape)

  def process_call(self, call_primitive, f, tracers, params):
    vals, polymorphic_shapes = unzip2((t.val, t.polymorphic_shape) for t in tracers)
    f, out_shapes_thunk = mask_subtrace(f, self.master, polymorphic_shapes)
    # TODO use call_primitive.bind(f, *vals, **params) instead here.
    #  currently breaks MaskingTest.test_where
    out_vals = f.call_wrapped(*vals)
    return map(partial(MaskTracer, self), out_vals, out_shapes_thunk())

  def post_process_call(self, call_primitive, out_tracers, params):
    vals, polymorphic_shapes = unzip2((t.val, t.polymorphic_shape) for t in out_tracers)
    master = self.master
    def todo(x):
      trace = MaskTrace(master, core.cur_sublevel())
      return map(partial(MaskTracer, trace), x, polymorphic_shapes)
    return vals, todo

polymorphic.polymorphic_trace_types.append(MaskTrace)