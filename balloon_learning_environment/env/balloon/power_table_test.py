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

"""Tests for power_table."""

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env.balloon import power_table


class PowerTableTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='low_pressure_ratio', pressure_ratio=0.98),
      dict(testcase_name='high_pressure_ratio', pressure_ratio=5.01))
  def test_invalid_pressure_ratios(self, pressure_ratio: float):
    with self.assertRaises(AssertionError):
      power_table.lookup(pressure_ratio, 1.0)

  @parameterized.parameters(
      (1.0, 0.2, 0),
      (1.0, 0.3, 150),
      (1.0, 0.4, 175),
      (1.0, 0.5, 200),
      (1.0, 0.6, 200),
      (1.08, 0.2, 0),
      (1.08, 0.3, 200),
      (1.08, 0.4, 200),
      (1.08, 0.7, 225),
      (1.08, 0.8, 225),
      (1.11, 0.2, 0),
      (1.11, 0.3, 225),
      (1.11, 0.4, 225),
      (1.11, 0.6, 250),
      (1.11, 0.7, 250),
      (1.14, 0.2, 0),
      (1.14, 0.3, 200),
      (1.14, 0.4, 225),
      (1.14, 0.5, 250),
      (1.14, 0.6, 250),
      (1.17, 0.2, 0),
      (1.17, 0.3, 225),
      (1.17, 0.4, 250),
      (1.17, 0.5, 275),
      (1.17, 0.6, 275),
      (1.2, 0.3, 0),
      (1.2, 0.4, 275),
      (1.2, 0.5, 300),
      (1.2, 0.6, 300),
      (1.23, 0.4, 0),
      (1.23, 0.5, 300),
      (1.23, 0.6, 325),
      (1.23, 0.7, 325),
      (1.26, 0.4, 0),
      (1.26, 0.5, 325),
      (1.26, 0.6, 350),
      (1.26, 0.7, 350))
  def test_table_lookup(self, pressure_ratio, state_of_charge,
                        expected_power_to_use):
    self.assertEqual(power_table.lookup(pressure_ratio, state_of_charge),
                     expected_power_to_use)


if __name__ == '__main__':
  absltest.main()
