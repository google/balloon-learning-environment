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

"""Tests for balloon_learning_environment.env.wind_field."""

import datetime as dt

from absl.testing import absltest
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.utils import units


class WindFieldTest(absltest.TestCase):

  def setUp(self):
    super(WindFieldTest, self).setUp()
    self.x = units.Distance(km=2.1)
    self.y = units.Distance(km=2.2)
    self.delta = dt.timedelta(minutes=3)

  def test_some_altitude_goes_north(self):
    field = wind_field.SimpleStaticWindField()
    self.assertEqual(
        wind_field.WindVector(
            units.Velocity(mps=0.0), units.Velocity(mps=10.0)),
        field.get_forecast(self.x, self.y, 9323.0, self.delta))

  def test_some_altitude_goes_south(self):
    field = wind_field.SimpleStaticWindField()
    self.assertEqual(
        wind_field.WindVector(
            units.Velocity(mps=0.0), units.Velocity(mps=-10.0)),
        field.get_forecast(self.x, self.y, 13999.0, self.delta))

  def test_some_altitude_goes_east(self):
    field = wind_field.SimpleStaticWindField()
    self.assertEqual(
        wind_field.WindVector(
            units.Velocity(mps=10.0), units.Velocity(mps=0.0)),
        field.get_forecast(self.x, self.y, 5523.0, self.delta))

  def test_some_altitude_goes_west(self):
    field = wind_field.SimpleStaticWindField()
    self.assertEqual(
        wind_field.WindVector(
            units.Velocity(mps=-10.0), units.Velocity(mps=0.0)),
        field.get_forecast(self.x, self.y, 11212.0, self.delta))

  def test_get_forecast_column_gives_same_result_as_get_forecast(self):
    field = wind_field.SimpleStaticWindField()
    forecast_10k = field.get_forecast(self.x, self.y, 10_000.0, self.delta)
    forecast_11k = field.get_forecast(self.x, self.y, 11_000.0, self.delta)
    forecast_column = field.get_forecast_column(
        self.x, self.y, [10_000.0, 11_000.0], self.delta)

    self.assertEqual(forecast_10k, forecast_column[0])
    self.assertEqual(forecast_11k, forecast_column[1])


if __name__ == '__main__':
  absltest.main()
