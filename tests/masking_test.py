# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from functools import partial
from unittest import SkipTest

import numpy as onp
from absl.testing import absltest, parameterized
from jax.api import _parse_shape_spec
from jax.interpreters.masking import shape_as_value
from jax import numpy as np, test_util as jtu, mask, vmap, jit, grad, lax, \
  ShapeError, core as jc, shapecheck, eval_polymorphic_shape, safe_map, \
  safe_zip, random
from jax.config import config
from jax.lax.lax import _identity
from jax.random import uniform, PRNGKey
from jax.scipy.special import expit
from operator import add, sub

config.parse_flags_with_absl()

map = safe_map
zip = safe_zip


# These are 'manual' tests for masking. The more exhaustive,
# more systematic tests should live in lax_test.py.

class MaskingTest(jtu.JaxTestCase):

  def test_sum(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(x)

    ans = padded_sum([np.array([3, 1, 4, 1, 5])], dict(n=3))
    expected = 8
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = padded_sum([np.array([3, 1, 4, 1, 5])], dict(n=4))
    expected = 9
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_sum_vmap(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(x)

    ans = vmap(padded_sum)([np.ones((5, 10))], dict(n=np.arange(5)))
    expected = onp.array([0, 1, 2, 3, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_add(self):
    # TODO:
    # self.check(add, ['(m, n)', 'n'], dict(m=3, n=3), '(m, n)')
    # self.check(add, ['n', ''], dict(n=3), 'n')
    # self.check(add, ['n', 'n'], dict(n=3), 'n')

    addvecs = mask(add, in_shapes=['n', 'n'], out_shape='n')

    x = np.array([3, 1, 4, 1, 5, 9])
    y = np.array([2, 6, 5, 3, 5, 8])
    ans = addvecs([x, y], dict(n=3))
    expected = onp.array([5, 7, 9])
    self.assertAllClose(ans[:3], expected, check_dtypes=False)

    thunk = lambda: addvecs([np.arange(5), np.arange(6)], dict(n=3))
    self.assertRaisesRegex(ShapeError, "", thunk)

  def test_scan(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    ans = cumsum([np.array([5, 2, 9, 1, 4])], dict(n=3))
    expected = 16
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_scan_vmap(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    ans = vmap(cumsum)([np.arange(6).reshape(2, 3)], dict(n=np.array([1, 2])))
    expected = onp.array([0, 7])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_scan_jit(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    @jit
    def jit_cumsum(args, shape_env):
      assert python_should_be_executing
      return cumsum(args, shape_env)

    python_should_be_executing = True
    ans = jit_cumsum([np.array([5, 2, 9, 1, 4])], dict(n=3))
    expected = 16
    self.assertAllClose(ans, expected, check_dtypes=False)

    python_should_be_executing = False
    ans = jit_cumsum([np.array([5, 2, 9, 1, 4])], dict(n=4))
    expected = 17
    self.assertAllClose(ans, expected, check_dtypes=False)

    python_should_be_executing = False
    ans = jit_cumsum([np.array([5, 2, 9, 1, 4])], dict(n=1))
    expected = 5
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_concatenate(self):
    @partial(mask, in_shapes=['n', 'm', 'n'], out_shape='m + 2 * n')
    def cat(x, y, z):
      return lax.concatenate([x, y, z], 0)

    ans = cat([np.array([1, 9]), np.array([2, 4, 9]), np.array([3, 9])],
              dict(n=1, m=2))
    expected = onp.array([1, 2, 4, 3])
    self.assertAllClose(ans[:4], expected, check_dtypes=False)

  def test_dot(self):
    @partial(mask, in_shapes=['(m, k)', '(k, n)'], out_shape='(m, n)')
    def dot(x, y):
      return lax.dot(x, y)

    x = onp.arange(6, dtype=onp.float32).reshape((2, 3))
    y = onp.arange(12, dtype=onp.float32).reshape((3, 4))
    ans = dot([x, y], dict(m=2, k=2, n=2))
    expected = onp.dot(x[:2, :2], y[:2, :2])
    self.assertAllClose(ans[:2, :2], expected, check_dtypes=False)

  def test_mean(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(x) / shape_as_value(x.shape)[0]

    ans = padded_sum([np.array([3, 1, 4, 1, 5])], dict(n=3))
    expected = 8 / 3
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='')
    def padded_sum(x):
      return np.sum(x)

    ans = padded_sum([np.array([[3, 4], [5, 6]])], dict(n=1))
    expected = 8
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic2(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='n')
    def padded_sum(x):
      return np.sum(x, axis=0)

    ans = padded_sum([np.array([[3, 4], [5, 6]])], dict(n=2))
    expected = np.array([8, 10])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic3(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='_')
    def padded_sum(x):
      return np.sum(x, axis=1)

    ans = padded_sum([np.array([[3, 4], [5, 6]])], dict(n=1))
    expected = np.array([3, 5])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_rnn(self):
    n = 3

    @partial(mask, in_shapes=['(_, _)', '(t, _)'], out_shape='_')
    def rnn(W, xs):
      def step(h, x):
        new_h = np.dot(W, h) + np.dot(W, x)
        return new_h, ()
      predicted, _ = lax.scan(step, np.zeros(n), xs)
      return predicted

    rng = onp.random.RandomState(0)
    W = np.eye(n)
    xs = rng.randn(10, n).astype(np.float_)
    ans = rnn([W, xs], dict(t=4))
    expected = xs[:4].sum(0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_rnn_grad(self):
    n = 3

    @partial(mask, in_shapes=['(_, _)', '(t, _)', '_'], out_shape='')
    def rnn(W, xs, target):
      def step(h, x):
        new_h = np.tanh(np.dot(W, h) + np.dot(W, x))
        return new_h, ()
      predicted, _ = lax.scan(step, np.zeros(n), xs)
      return np.sum((predicted - target)**2)

    rng = onp.random.RandomState(0)
    W = rng.randn(n, n).astype(np.float_)
    xs = rng.randn(10, n).astype(np.float_)
    y = rng.randn(n).astype(np.float_)

    ans = grad(lambda W: rnn([W, xs, y], dict(t=4)))(W)

    def rnn_reference(W, xs, target):
      h = np.zeros(n)
      for x in xs:
        h = np.tanh(np.dot(W, h) + np.dot(W, x))
      predicted = h
      return np.sum((predicted - target)**2)

    expected = grad(lambda W: rnn_reference(W, xs[:4], y))(W)

    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_ragged_batched_rnn(self):
    n = 3

    @partial(mask, in_shapes=('(_, _)', '(t, _)', '_'), out_shape='')
    def rnn(W, xs, target):
      def step(h, x):
        new_h = np.tanh(np.dot(W, h) + np.dot(W, x))
        return new_h, ()
      predicted, _ = lax.scan(step, np.zeros(n), xs)
      return np.sum((predicted - target)**2)

    rng = onp.random.RandomState(0)
    W = rng.randn(n, n).astype(np.float_)
    seqs = rng.randn(3, 10, n).astype(np.float_)
    ts = np.array([2, 5, 4])
    ys = rng.randn(3, n)

    ans = grad(lambda W: vmap(rnn, ((None, 0, 0), 0))((W, seqs, ys), dict(t=ts)).sum())(W)

    def rnn_reference(W, seqs, targets):
      total_loss = np.array(0, np.float_)
      for xs, target in zip(seqs, targets):
        h = np.zeros(n)
        for x in xs:
          h = np.tanh(np.dot(W, h) + np.dot(W, x))
        predicted = h
        total_loss = total_loss + np.sum((predicted - target)**2)
      return total_loss

    seqs_ = [xs[:t] for xs, t in zip(seqs, ts)]
    expected = grad(lambda W: rnn_reference(W, seqs_, ys).sum())(W)

    self.assertAllClose(
        ans, expected, check_dtypes=False,
        rtol=2e-2 if jtu.device_under_test() == "tpu" else 1e-5)

  def test_jit(self):
    self.check(jit(lambda x: lax.concatenate([x, x], 0)), ['n'], dict(n=3), '2*n')

  def test_device_put(self):
    self.check(lambda x: np.device_put(x), ['n'], dict(n=3), 'n')

  def check(self, fun, input_shapes, values_dict,
            out_shape=None, pad_dict=None, custom_inputs=None):
    if out_shape is not None:
      shapecheck(input_shapes, out_shape)(fun)

    masked_fun = mask(fun, input_shapes, out_shape)

    all_pad_dict = defaultdict(lambda: 3)
    if pad_dict is not None:
      all_pad_dict.update(pad_dict)

    padded_values_dict = {k: values_dict[k] + all_pad_dict[k] for k in values_dict.keys()}

    input_shapes = map(_parse_shape_spec, input_shapes)
    concrete_shapes = map(
      partial(eval_polymorphic_shape, values_dict=values_dict), input_shapes)
    inputs = list(map(partial(uniform, PRNGKey(0)), concrete_shapes))

    if custom_inputs is not None:
      for index, value in custom_inputs.items():
        inputs[index] = value

    padded_input_shapes = map(partial(eval_polymorphic_shape,
                                      values_dict=padded_values_dict), input_shapes)

    pad_widths = map(sub, map(onp.array, padded_input_shapes), concrete_shapes)
    padded_inputs = list(map(lambda input, widths: np.pad(input, tuple((0, w) for w in widths), constant_values=-1), inputs, pad_widths))
    out_ = fun(*inputs)
    padded_out = masked_fun(padded_inputs, values_dict)
    out = padded_out[tuple(slice(None, k) for k in out_.shape)]
    self.assertAllClose(out_, out, check_dtypes=True)

  @parameterized.named_parameters({
                                    'testcase_name': "padding_config={}_shapes={}".format(
                                      padding_config, shape),
                                    'padding_config': padding_config,
                                    'shape': shape}
                                  for padding_config, shape in (
                                          (((1, 2, 0),), (2,)),
                                          (((1, 2, 0), (3, 4, 0)), (1, 2)),
                                          (((0, 0, 0), (0, 0, 0)), (1, 2)),
                                          (((1, 2, 3),), (2,)),
                                          (((1, 2, 1), (3, 4, 2)), (3, 2)),
                                          (((-1, 2, 0),), (2,)),
                                          (((-1, -2, 0), (1, 2, 0)), (4, 2)),
                                          (((-1, 2, 0), (1, 2, 2)), (4, 2)),
                                          (((-1, -2, 2),), (5,)),
                                          (((-1, -2, 1), (1, 2, 2)), (4, 2))))
  def test_pad(self, padding_config, shape):
    def pad(x):
      return lax.pad(x, np.array(1., x.dtype), padding_config)

    if len(shape) == 1:
      self.check(pad, ['n'], dict(n=shape[0]))
    else:
      self.check(pad, ['(m,n)'], dict(m=shape[0], n=shape[1]))

  def test_numpy_pad(self):
    def numpy_pad(x):
      return np.pad(x, (0, 1), constant_values=np.array(5., x.dtype))

    self.check(numpy_pad, ['n'], dict(n=3), 'n+1')

  @parameterized.named_parameters(jtu.cases_from_list(
    {
      'testcase_name': "strides={}_padding={}_lhs_dilation={}_dimension_numbers"
                       "={}_lhs_perm={}_rhs_perm={}_out_perm={}".format(
        strides, padding, lhs_dilation, dimension_numbers, lhs_perm, rhs_perm, out_perm),
      'strides': strides, 'padding': padding, 'lhs_dilation': lhs_dilation,
      'dimension_numbers': dimension_numbers, 'lhs_perm': lhs_perm,
      'rhs_perm': rhs_perm, 'out_perm': out_perm}
    for strides in [(1, 1), (2, 1)]
    for padding in ['SAME', 'VALID', ((0, 1), (2, 0))]
    for lhs_dilation in (None, (1, 2))
    for dimension_numbers, (lhs_perm, rhs_perm, out_perm) in (
            (("NCHW", "OIHW", "NCHW"), ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))),
            (("NHWC", "HWIO", "NHWC"), ((0, 2, 3, 1), (2, 3, 1, 0), (0, 2, 3, 1))),
            (("NCHW", "HWIO", "NHWC"), ((0, 1, 2, 3), (2, 3, 1, 0), (0, 2, 3, 1)))
    )
    # String padding is not implemented for transposed convolution, see conv_general_dilated implementation:
    if (lhs_dilation is None or not isinstance(padding, str)) and
    # only test strides with same padding:
    (strides[0] == 1 or padding == 'SAME')))
  def test_conv(self, strides, padding, lhs_dilation,
                dimension_numbers, lhs_perm, rhs_perm, out_perm):
    valid = padding == 'VALID'
    is_strided = strides[0] != 1
    lhs_shape = '({}, {}, {}, {})'.format(*onp.take(['n', 'i', '2*h' if is_strided else 'h', 'w'], lhs_perm))
    rhs_shape = '({}, {}, {}, {})'.format(*onp.take(['o', 'i', '2', '3'], rhs_perm))
    out_shape = '({}, {}, {}, {})'.format(*onp.take([
      'n', 'o', 'h+-1' if valid and not is_strided else 'h',
      ('w+-2' if valid else 'w') if lhs_dilation is None else '2*w+-1'], out_perm))

    def conv(lhs, rhs):
      return lax.conv_general_dilated(
        lhs, rhs, strides, padding,
        lhs_dilation=lhs_dilation, dimension_numbers=dimension_numbers)

    self.check(conv, [lhs_shape, rhs_shape], dict(n=1, i=3, o=2, h=1, w=2),
               out_shape, pad_dict=dict(n=0, i=0, o=0))

  def test_indexing(self):
    self.check(lambda x: x[0], ['n'], dict(n=3), '')
    self.check(lambda x: x[-1], ['n'], dict(n=3), '')
    self.check(lambda x: x[..., -1], ['(n,a)'], dict(n=3, a=3), 'n')

  def test_slicing(self):
    self.check(lambda x: x[1:], ['n'], dict(n=3), 'n+-1')
    self.check(lambda x: x[:-1], ['n'], dict(n=3), 'n+-1')

  def test_reshape(self):
    self.check(lambda x: np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])),
               ['n, a, b'], dict(n=1, a=2, b=3), 'n, a*b',
               pad_dict=dict(a=0, b=0))

    message = "Reshaped dimensions have to be non-padded, so that logical and padded shapes match. " \
              "This error is currently also raised when reshaped dimensions are not at the end, a case which is not yet implemented."

    self.assertRaisesWithLiteralMatch(
      ValueError, message,
      lambda: self.check(lambda x: x.ravel(), ['(n,m)'], dict(n=2, m=2), 'n*m'))
    self.assertRaisesWithLiteralMatch(
      ValueError, message,
      lambda: self.check(lambda x: np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2])),
                         ['a, b, n'], dict(n=1, a=2, b=3), 'a*b, n',
                         pad_dict=dict(a=0, b=0)))

  def test_unsupported_op(self):
    p = jc.Primitive('unsupported_op')
    p.def_abstract_eval(_identity)
    p.def_impl(lambda x: x)

    @partial(mask, in_shapes=['n'], out_shape='n')
    def unsupported(x):
      return p.bind(x)

    def thunk():
      unsupported([np.zeros((1, ))], dict(n=1))

    self.assertRaisesWithLiteralMatch(NotImplementedError, "Masking rule for unsupported_op not implemented yet.", thunk)

  def test_wrong_out_shape(self):
    masked = mask(lambda x: x, in_shapes=['n'], out_shape='n+-1')

    def thunk():
      masked([np.zeros((1, ))], dict(n=1))

    self.assertRaisesWithLiteralMatch(ShapeError, "Output shapes should be [(n + -1,)] but are [(n,)].", thunk)

  def test_nesting(self):
    raise SkipTest("not yet implemented")

    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(x)

    batched_sum = vmap(padded_sum)

    @partial(mask, in_shapes=['(m, _)', 'm'], out_shape='')
    def fun(x, ns):
      return batched_sum([x], dict(n=ns)).sum()

    x = np.array([[3, 1, 4, 1],
                  [5, 9, 2, 6],
                  [5, 3, 5, 8]])
    ns = np.array([2, 3, 2])
    ans = fun([x, ns], dict(m=2))
    expected = 3+1 + 5+9+2
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_arange(self):
    raise SkipTest("not yet implemented")

    @partial(mask, in_shapes=['n'], out_shape='n')
    def padded_add(x):
      return x + lax.iota(x.shape[0])

    ans = padded_add([np.array([3, 1, 4, 1, 5])], dict(n=3))
    expected = onp.array([3, 2, 6])
    self.assertAllClose(ans[:3], expected, check_dtypes=False)

  def test_sum_2d(self):
    self.check(lambda x: np.sum(x), ['(m, n)'], dict(m=3, n=3), '')

  def test_expit(self):
    self.check(lambda x: expit(x), ['n'], dict(n=3), 'n')

  def test_uniform(self):
    raise SkipTest

    self.check(lambda key, x: random.uniform(key, x.shape), ['2', 'n'], dict(n=3), 'n', custom_inputs={0: PRNGKey(0)})

if __name__ == '__main__':
  absltest.main()
