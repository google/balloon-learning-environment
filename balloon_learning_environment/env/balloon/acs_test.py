# coding=utf-8
# Copyright 2021 The Balloon Learning Environment Authors.
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

"""Tests for balloon_learning_environment.env.acs."""

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env.balloon import acs
from balloon_learning_environment.utils import units


class AcsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='at_min', pressure_ratio=1.0, power=100.0,
           comparator='eq'),
      dict(testcase_name='at_mid', pressure_ratio=1.2, power=300.0,
           comparator='eq'),
      dict(testcase_name='at_max', pressure_ratio=1.35, power=400.0,
           comparator='eq'),
      dict(testcase_name='below_min', pressure_ratio=0.01, power=100.0,
           comparator='lt'),
      dict(testcase_name='above_max', pressure_ratio=2.0, power=400.0,
           comparator='gt'))
  def test_get_most_efficient_power(self, pressure_ratio, power, comparator):
    if comparator == 'eq':
      comparator = self.assertEqual
    elif comparator == 'lt':
      comparator = self.assertLessEqual
    else:
      comparator = self.assertGreaterEqual
    comparator(acs.get_most_efficient_power(pressure_ratio).watts, power)

  @parameterized.named_parameters(
      dict(testcase_name='at_min', pressure_ratio=1.05, power=100.0,
           efficiency=0.4, comparator='eq'),
      dict(testcase_name='at_max', pressure_ratio=1.35, power=400.0,
           efficiency=0.13, comparator='eq'),
      dict(testcase_name='below_min', pressure_ratio=0.01, power=10.0,
           efficiency=0.4, comparator='gt'),
      dict(testcase_name='above_max', pressure_ratio=2.0, power=500.0,
           efficiency=0.13, comparator='lt'))
  def test_get_fan_efficiency(self, pressure_ratio, power, efficiency,
                              comparator):
    if comparator == 'eq':
      comparator = self.assertEqual
    elif comparator == 'lt':
      comparator = self.assertLessEqual
    else:
      comparator = self.assertGreaterEqual
    comparator(acs.get_fan_efficiency(pressure_ratio, units.Power(watts=power)),
               efficiency)

  def test_get_mass_flow(self):
    self.assertEqual(
        acs.get_mass_flow(units.Power(watts=3.6), 10.0), 0.01)


if __name__ == '__main__':
  absltest.main()
