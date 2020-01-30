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

import numpy as onp
from absl.testing import absltest, parameterized

from jax import numpy as np, test_util as jtu, lax, random, vmap, shapecheck, \
  jit
from jax.abstract_arrays import Poly, Mon
from jax.api import _parse_shape_spec, ShapeError, _remap_ids, _UniqueIds
from jax.random import PRNGKey
from jax.config import config
config.parse_flags_with_absl()


# These are 'manual' tests for shape checking. The more exhaustive,
# more systematic tests should live in lax_test.py.

def const_poly(c):
  return Poly({Mon(): c})

class PolymorphicTest(jtu.JaxTestCase):

  @parameterized.parameters([
      ['(m, n)', 'ShapeSpec(m, n)'],
      ['(m * n)', 'ShapeSpec(m n)'],
      ['m * n', 'ShapeSpec(m n)'],
      ['(m * n,)', 'ShapeSpec(m n)'],
      ['(3, m)', 'ShapeSpec(3, m)'],
      ['(10, m)', 'ShapeSpec(10, m)'],
      ['(-10, m)', 'ShapeSpec(-10, m)'],
      ['(3 * m)', 'ShapeSpec(3 m)'],
      ['m', 'ShapeSpec(m)'],
      ['', 'ShapeSpec()'],
      ['n + -1*n', 'ShapeSpec(0)'],
      ['m + n', 'ShapeSpec(m + n)'],
      ['m + n * k', 'ShapeSpec(k n + m)'],
      ['m + 3 * k', 'ShapeSpec(3 k + m)'],
      ['-3 + k + k * k', 'ShapeSpec(k**2 + k + -3)'],
      ['', 'ShapeSpec()'],
      ['_', 'ShapeSpec(_)'],
  ])
  def test_parse_spec(self, spec, ans):
    self.assertEqual(str(_parse_shape_spec(spec)), ans)
    self.assertEqual(str(_remap_ids(_UniqueIds(), _parse_shape_spec(spec))), ans)

  def test_Poly_equal(self):
    assert const_poly(3) == 3
    assert onp.array(3, onp.int64) == const_poly(3)
    assert onp.array(3, onp.int64)[()] == const_poly(3)
    assert not onp.array(3, onp.int64) != const_poly(3)
    assert const_poly(4) != 3
    assert 3 == const_poly(3)
    assert 4 != const_poly(3)
    assert const_poly(4) == const_poly(4)
    assert const_poly(3) != const_poly(4)
    assert Poly({Mon(): 3, Mon({'n': 1}): 4}) == Poly({Mon({'n': 1}): 4, Mon(): 3})
    assert Poly({Mon(): 3, Mon({'n': 1}): 4}) != Poly({Mon(): 3, Mon({'n': 2}): 4})
    assert Poly({Mon(): 3, Mon({'m': 1}): 4}) != Poly({Mon(): 3, Mon({'n': 1}): 4})

  def test_Poly_hash(self):
    assert not len(set(hash(Poly({Mon(): i})) for i in range(10))) == 1
    assert hash(Poly({Mon(): 3, Mon({'n': 1}): 4})) == hash(Poly({Mon({'n': 1}): 4, Mon(): 3}))

  def test_Mon_hash(self):
    assert not len(set(hash(Mon({'a': i})) for i in range(10))) == 1
    assert hash(Mon({'a': 1, 'b': 1})) == hash(Mon({'b': 1, 'a': 1}))

  def test_Poly_compare(self):
    poly = Poly({Mon(): 3, Mon({'n': 1}): 4})
    # Assume poly > 0 to make various shape rules work with polymorphic shapes:
    assert poly >= 0
    assert poly >= 1
    assert poly > 0

    assert 0 <= poly
    assert 0 < poly
    assert const_poly(3) >= 1
    assert const_poly(3) > 1
    self.assertRaisesRegex(ValueError, "", lambda: poly >= 2)
    self.assertRaisesRegex(ValueError, "", lambda: poly > 1)

  def test_Poly_divmod(self):
    n = Poly({Mon({'n': 1}): 1})
    assert (n, 1) == divmod(2*n+1, 2)
    assert (2*n, 0) == divmod(10*n, 5)
    assert (2*n+4, 3) == divmod(10*n+23, 5)

if __name__ == '__main__':
  absltest.main()
