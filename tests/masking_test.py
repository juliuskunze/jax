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
import scipy.stats

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
    self.check(add, ['(m, n)', 'n'], dict(m=3, n=3), '(m, n)', unpadded_vars=['n'])
    self.check(add, ['n', ''], dict(n=3), 'n')
    self.check(add, ['n', 'n'], dict(n=3), 'n')

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
    self.check(lambda x, y, z: lax.concatenate([x, y, z], 0),
               ['n', 'm', 'n'], dict(n=1, m=2), 'm + 2 * n')

  def test_output_shape_error(self):
    def thunk(skip_shapecheck=False):
      self.check(lambda x: x, ['n'], dict(n=2), 'n+-1')

    message = "Output shapes should be (n + -1,) but are (n,)."
    self.assertRaisesWithLiteralMatch(ShapeError, message, thunk)
    self.assertRaisesWithLiteralMatch(ShapeError, message, partial(thunk, skip_shapecheck=True))

    def thunk(skip_shapecheck=False):
      self.check(lambda x: np.split(x, 2),
                 ['2*n'], dict(n=3), ['7*n', 'n'], unpadded_vars=['n'],
                 skip_shapecheck=skip_shapecheck)

    message = "Output shapes should be [(7 n,), (n,)] but are [(n,), (n,)]."
    self.assertRaisesWithLiteralMatch(ShapeError, message, thunk)
    self.assertRaisesWithLiteralMatch(ShapeError, message, partial(thunk, skip_shapecheck=True))

  def test_output_tree_error(self):
    def thunk(skip_shapecheck=False):
      self.check(lambda x: np.split(x, 2), ['2*n'], dict(n=3), ('n', 'n'), unpadded_vars=['n'],
                 skip_shapecheck=skip_shapecheck)
    message = "Output shapes should be ((n,), (n,)) but are [(n,), (n,)]."
    self.assertRaisesWithLiteralMatch(ShapeError, message, thunk)
    self.assertRaisesWithLiteralMatch(ShapeError, message, partial(thunk, skip_shapecheck=True))

  def test_dot(self):
    self.check(lambda x, y: lax.dot(x, y),
               ['(m, k)', '(k, n)'], dict(m=2, k=2, n=2), '(m, n)')
    self.check(lambda A, b: np.dot(A, b), ['(m, n)', 'n'], dict(m=2, n=2), 'm')

    def thunk():
      self.check(lambda A, b: lax.dot_general(A, b, [((0,), (0,)), ((), ())]),
                 ['(m, n)', 'n'], dict(m=2, n=2), 'm')
    self.assertRaisesRegex(TypeError, "", thunk)

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
    # TODO: check fun is actually jitted
    self.check(jit(lambda x: lax.concatenate([x, x], 0)), ['n'], dict(n=3), '2*n')

  def test_device_put(self):
    self.check(lambda x: np.device_put(x), ['n'], dict(n=3), 'n')

  def check(self, fun, input_shapes, values_dict,
            out_shape=None, unpadded_vars=None, custom_inputs=None,
            skip_shapecheck=False, check_output=None):
    if out_shape is not None and not skip_shapecheck:
      shapecheck(input_shapes, out_shape)(fun)

    masked_fun = mask(fun, input_shapes, out_shape)

    pad_dict = defaultdict(lambda: 2)
    if unpadded_vars is not None:
      for var in unpadded_vars:
        pad_dict[var] = 0

    padded_values_dict = {k: values_dict[k] + pad_dict[k] for k in values_dict.keys()}

    input_shapes = map(_parse_shape_spec, input_shapes)
    concrete_shapes = map(
      partial(eval_polymorphic_shape, values_dict=values_dict), input_shapes)
    inputs = list(map(partial(uniform, PRNGKey(0)), concrete_shapes))

    if custom_inputs is not None:
      for index, value in custom_inputs.items():
        inputs[index] = value

    padded_input_shapes = map(partial(eval_polymorphic_shape,
                                      values_dict=padded_values_dict), input_shapes)

    pad_widths = map(sub, map(partial(onp.array, dtype=onp.int64), padded_input_shapes), concrete_shapes)
    padded_inputs = [np.pad(input, tuple((0, w) for w in widths), constant_values=-1) if input.ndim > 0 else input
                     for input, widths in zip(inputs, pad_widths)]
    out_ = fun(*inputs)
    padded_out = masked_fun(padded_inputs, values_dict)

    def check_padded_output(out_, padded_out):
      out = padded_out[tuple(slice(None, k) for k in out_.shape)]

      if check_output:
        check_output(out_, out)
      else:
        self.assertAllClose(out_, out, check_dtypes=True)

    assert type(out_) == type(padded_out)
    if type(padded_out) in (tuple, list):
      map(check_padded_output, out_, padded_out)
    else:
      check_padded_output(out_, padded_out)

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

  def test_pad_check_out_shape(self):
    self.check(lambda x: lax.pad(x, np.array(0., x.dtype), [(1, 1, 1)]),
               ['n'], dict(n=3), '2*n+1')

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
               out_shape, unpadded_vars=['n', 'i', 'o'])

  def test_indexing(self):
    self.check(lambda x: x[0], ['n'], dict(n=3), '')
    self.check(lambda x: x[-1], ['n'], dict(n=3), '')
    self.check(lambda x: x[..., -1], ['(n,a)'], dict(n=3, a=3), 'n')

  def  test_slicing(self):
    self.check(lambda x: x[1:], ['n'], dict(n=3), 'n+-1')
    self.check(lambda x: x[:-1], ['n'], dict(n=3), 'n+-1')
    self.check(lambda x: x[..., -1], ['(n,a)'], dict(n=3, a=3), 'n')

  def test_lax_slice(self):
    self.check(lambda x: lax.slice(x, (1,), (x.shape[0],)), ['n'], dict(n=3), 'n+-1')
    # TODO: self.check(lambda x: lax.slice(x, (x.shape[0] // 2,), (x.shape[0],)), ['2*n'], dict(n=3), 'n')

  def test_reshape(self):
    self.check(lambda x: np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])),
               ['n, a, b'], dict(n=1, a=2, b=3), 'n, a*b',
               unpadded_vars=['a', 'b'])

    def check_shapes_match(out_, out):
      self.assertEqual(out_.shape, out.shape)

    # Only check for shapes in case of reshaping padded dimensions.
    # Needed for random number generation:
    self.check(lambda x: x.ravel(), ['(n,m)'], dict(n=2, m=2), 'n*m',
               check_output=check_shapes_match)
    self.check(lambda x: np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2])),
               ['a, b, n'], dict(n=1, a=2, b=3), 'a*b, n',
               check_output=check_shapes_match)

  def test_transpose(self):
    self.check(lambda x: np.transpose(x, (1, 0, 2)),
               ['(a, b, c)'], dict(a=1, b=2, c=3), 'b, a, c')

  def test_unsupported_op(self):
    p = jc.Primitive('unsupported_op')
    p.def_abstract_eval(_identity)
    p.def_impl(lambda x: x)

    def thunk():
      self.check(lambda x: p.bind(x), ['n'], dict(n=1), 'n')

    message = "Masking rule for unsupported_op not implemented yet."
    self.assertRaisesWithLiteralMatch(NotImplementedError, message, thunk)

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
    self.check(lambda x: -np.arange(x.shape[0]), ['n'], dict(n=3), 'n')

  def test_sum_2d(self):
    self.check(lambda x: np.sum(x), ['(m, n)'], dict(m=3, n=3), '')

  def test_expit(self):
    self.check(lambda x: expit(x), ['n'], dict(n=3), 'n')

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
    for dtype in [onp.float32, onp.float64]))
  def test_uniform(self, dtype):
    # TODO: how to allow input shape `n`?
    #  random.threefry_2x32 handles even and odd sizes differently,
    #  making general size `n` fail.

    @shapecheck(['2', '2*n+1'], '2*n+1')
    @shapecheck(['2', '2*n'], '2*n')
    def sample(key, x):
      # TODO remove workaround by allow specifying types in type specs:
      key = key.astype(onp.uint64)
      return random.uniform(key, x.shape, dtype)

    # TODO currently random.uniform(k, (2,)) != random.uniform(k, (3,))[:2]
    def check_output(expected_out, out):
      assert expected_out.shape == out.shape
      fail_prob = 0.01  # conservative bound on statistical fail prob by Kolmo CDF
      self.assertGreater(scipy.stats.kstest(out, scipy.stats.uniform().cdf).pvalue, fail_prob)

    self.check(sample, ['2', '2*n'], dict(n=10000), '2*n',
               custom_inputs={0: PRNGKey(0)},
               check_output=check_output)

  def test_zeros(self):
    self.check(lambda x: -np.zeros(x.shape), ['n'], dict(n=3), 'n')

  def test_ones(self):
    self.check(lambda x: -np.ones(x.shape), ['n'], dict(n=3), 'n')

  def test_broadcast_to(self):
    self.check(lambda x: -np.broadcast_to(0, x.shape), ['n'], dict(n=3), 'n')

  def test_broadcast_in_dim(self):
    self.check(lambda x: -lax.broadcast_in_dim(np.zeros((1, 1)), shape=(3, x.shape[0], 4), broadcast_dimensions=(1, 2)),
               ['(n, 1)'], dict(n=3), '(3, n, 4)')

  def test_destructure(self):
    def d(key):
      key1, key2 = key
      return key1

    self.check(d, ['2'], dict(), '')

  def test_where(self):
    self.check(lambda x: -np.where(x < 0, x, 0.), ['n'], dict(n=3), 'n')
    # TODO: self.check(lambda x: -np.where(x < 0, x, np.zeros_like(x)), ['n'], dict(n=3), 'n')

  def test_split(self):
    self.check(lambda x: np.split(x, 2), ['2*n'], dict(n=3), ['n', 'n'], unpadded_vars=['n'])
    self.check(lambda x: np.split(x, [10]), ['n'], dict(n=15), ['10', 'n+-10'], unpadded_vars=['n'])

  @parameterized.named_parameters(jtu.cases_from_list([{
    'testcase_name': "operator={}".format(operator.__name__), 'operator': operator}
    for operator in [np.sum, np.prod, np.max, np.min]]))
  def test_reduce(self, operator):
    self.check(operator, ['(m, n)'], dict(m=3, n=3), '', unpadded_vars=['m', 'n'])

if __name__ == '__main__':
  absltest.main()
