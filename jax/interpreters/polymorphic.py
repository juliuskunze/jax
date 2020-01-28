from jax.interpreters.partial_eval import PartialVal, StagingJaxprTrace, \
  JaxprTrace, tracers_to_jaxpr, instantiate_const_at

from .. import core
from .. import linear_util as lu
from ..util import unzip2, safe_zip, safe_map, partial
from ..core import new_master, AbstractValue, unit, Tracer

map = safe_map
zip = safe_zip

# TODO can we avoid duplicating the functions below,
#  i. e. by attaching aux info to JaxprTrace
#  within pe.trace_to_subjaxpr instead of subtyping?

class PolymorphicJaxprTrace(JaxprTrace): pass

@lu.transformation
def _trace_to_subjaxpr(master, instantiate, pvals):
  """Copy of trace_to_subjaxpr, but uses PolymorphicJaxprTrace."""
  assert all([isinstance(pv, PartialVal) for pv in pvals]), pvals
  trace = PolymorphicJaxprTrace(master, core.cur_sublevel())
  in_tracers = map(trace.new_arg, pvals)
  ans = yield in_tracers, {}
  instantiate = [instantiate] * len(ans) if type(instantiate) is bool else instantiate
  out_tracers = map(trace.full_raise, map(core.full_lower, ans))
  out_tracers = map(partial(instantiate_const_at, trace), instantiate, out_tracers)
  jaxpr, consts, env = tracers_to_jaxpr(in_tracers, out_tracers)
  out_pvals = [t.pval for t in out_tracers]
  del trace, in_tracers, out_tracers
  yield jaxpr, (out_pvals, consts, env)

def _trace_to_jaxpr(fun, pvals, instantiate=False):
  """Copy of trace_to_jaxpr, but uses PolymorphicJaxprTrace."""
  with new_master(PolymorphicJaxprTrace) as master:
    fun = _trace_to_subjaxpr(fun, master, instantiate)
    jaxpr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
    assert not env
    del master

  return jaxpr, out_pvals, consts

def abstract_eval_fun(fun, *avals, **params):
  """Copy of abstract_eval_fun, , but uses PolymorphicJaxprTrace."""
  pvals_in = [PartialVal((a, unit)) for a in avals]
  _, pvals_out, _ = _trace_to_jaxpr(lu.wrap_init(fun, params), pvals_in,
                                    instantiate=True)
  avals_out, _ = unzip2(pvals_out)
  for aval_out in avals_out:
    assert isinstance(aval_out, AbstractValue)  # instantiate=True
  return avals_out

def trace_to_jaxpr_from_avals(fun, *avals, **params):
  pvals_in = [PartialVal((a, unit)) for a in avals]
  jaxpr, _, _ = _trace_to_jaxpr(lu.wrap_init(fun, params), pvals_in,
                                    instantiate=True)
  return jaxpr

polymorphic_trace_types = [PolymorphicJaxprTrace]

def ensure_traced(operand):
  if isinstance(operand, Tracer):
    return operand

  def has_poly_trace(master):
    return issubclass(master.trace_type, tuple(polymorphic_trace_types))

  masters = reversed(core.trace_state.trace_stack.upward)
  master = next(filter(has_poly_trace, masters))
  trace = master.trace_type(master, core.cur_sublevel())
  return trace.pure(operand)