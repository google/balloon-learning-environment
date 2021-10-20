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

"""Tests for pressure_range_builder."""

import functools

from absl.testing import absltest
from balloon_learning_environment.env.balloon import altitude_safety
from balloon_learning_environment.env.balloon import pressure_range_builder
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import test_helpers
import jax


class AltitudeRangeBuilderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))
    self.create_balloon = functools.partial(
        test_helpers.create_balloon, atmosphere=self.atmosphere)

  def test_get_pressure_range_returns_valid_range(self):
    b = self.create_balloon()

    pressure_range = pressure_range_builder.get_pressure_range(
        b.state, self.atmosphere)

    self.assertIsInstance(pressure_range,
                          pressure_range_builder.AccessiblePressureRange)
    self.assertBetween(pressure_range.min_pressure, 1000.0, 100_000.0)
    self.assertBetween(pressure_range.max_pressure, 1000.0, 100_000.0)

  def test_get_pressure_range_returns_min_pressure_below_max_pressure(self):
    b = self.create_balloon()

    pressure_range = pressure_range_builder.get_pressure_range(
        b.state, self.atmosphere)

    self.assertLess(pressure_range.min_pressure, pressure_range.max_pressure)

  def test_get_pressure_range_returns_max_pressure_above_min_altitude(self):
    b = self.create_balloon()

    pressure_range = pressure_range_builder.get_pressure_range(
        b.state, self.atmosphere)

    self.assertLessEqual(
        pressure_range.max_pressure,
        self.atmosphere.at_height(altitude_safety.MIN_ALTITUDE).pressure)

  # TODO(joshgreaves): Add more tests when the pressure ranges are as expected.


if __name__ == '__main__':
  absltest.main()
