# coding=utf-8
# Copyright 2022 The Balloon Learning Environment Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for transforms."""

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.utils import transforms


class TransformsTest(parameterized.TestCase):

  def test_linear_rescale_with_extrapolation_with_invalid_range(self):
    with self.assertRaises(ValueError):
      transforms.linear_rescale_with_extrapolation(1.0, 1.0, 0.0)

  @parameterized.named_parameters(
      dict(testcase_name='Interpolate', x=15.0, expected=0.5),
      dict(testcase_name='LowerLimit', x=10.0, expected=0.0),
      dict(testcase_name='UpperLimit', x=20.0, expected=1.0),
      dict(testcase_name='LowerExtrapolate', x=5.0, expected=-0.5),
      dict(testcase_name='UpperExtrapolate', x=25.0, expected=1.5))
  def test_linear_rescale_with_extrapolation(self, x: float, expected: float):
    self.assertEqual(
        transforms.linear_rescale_with_extrapolation(x, 10., 20.),
        expected)

  def test_linear_rescale_with_saturation_with_invalid_range(self):
    with self.assertRaises(ValueError):
      transforms.linear_rescale_with_saturation(1.0, 1.0, 0.0)

  @parameterized.named_parameters(
      dict(testcase_name='Interpolate', x=15.0),
      dict(testcase_name='LowerLimit', x=10.0),
      dict(testcase_name='UpperLimit', x=20.0),
      dict(testcase_name='LowerExtrapolate', x=5.0),
      dict(testcase_name='UpperExtrapolate', x=25.0))
  def test_undo_linear_rescale_with_extrapolation(self, x: float):
    rescaled = transforms.linear_rescale_with_extrapolation(x, 10.0, 20.0)
    unrescaled = transforms.undo_linear_rescale_with_extrapolation(
        rescaled, 10.0, 20.0)
    self.assertAlmostEqual(x, unrescaled)

  @parameterized.named_parameters(
      dict(testcase_name='Interpolate', x=15.0, expected=0.5),
      dict(testcase_name='LowerLimit', x=10.0, expected=0.0),
      dict(testcase_name='UpperLimit', x=20.0, expected=1.0),
      dict(testcase_name='LowerCapped', x=5.0, expected=0.0),
      dict(testcase_name='UpperCapped', x=25.0, expected=1.0))
  def test_linear_rescale_with_saturation(self, x: float, expected: float):
    self.assertEqual(
        transforms.linear_rescale_with_saturation(x, 10.0, 20.0),
        expected)

  def test_squash_to_unit_interval_with_invalid_constant(self):
    with self.assertRaises(ValueError):
      transforms.squash_to_unit_interval(1., -1.)

  @parameterized.named_parameters(
      dict(testcase_name='ZeroX', x=0.0, expected=0.0),
      dict(testcase_name='OneX', x=1.0, expected=1.0),
      dict(testcase_name='LargeX', x=500.0, expected=1.0))
  def test_squash_to_unit_interval(self, x: float, expected: float):
    self.assertAlmostEqual(transforms.squash_to_unit_interval(x, 1e-9),
                           expected)

  @parameterized.named_parameters(
      dict(testcase_name='SimpleValue', x=10.0),
      dict(testcase_name='ZeroX', x=0.0), dict(testcase_name='LargeX', x=500.0))
  def test_undo_squash_to_unit_interval(self, x: float):
    squashed = transforms.squash_to_unit_interval(x, 1.0)
    unsquashed = transforms.undo_squash_to_unit_interval(squashed, 1.0)
    self.assertAlmostEqual(unsquashed, x)


if __name__ == '__main__':
  absltest.main()
