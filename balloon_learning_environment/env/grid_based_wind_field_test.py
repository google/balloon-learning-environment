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

"""Tests for grid_based_wind_field."""

import datetime as dt

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env import grid_based_wind_field
from balloon_learning_environment.env import grid_wind_field_sampler
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.generative import vae
from balloon_learning_environment.utils import test_helpers
from balloon_learning_environment.utils import units
import jax
from jax import numpy as jnp
import numpy as np


class SimpleWindFieldSampler(grid_wind_field_sampler.GridWindFieldSampler):

  @property
  def field_shape(self) -> vae.FieldShape:
    return vae.FieldShape()

  def sample_field(self,
                   key: jnp.ndarray,
                   date_time: dt.datetime) -> np.ndarray:
    return np.asarray(
        jax.random.normal(key,
                          vae.FieldShape().grid_shape(),
                          dtype=jnp.float32))


class GridBasedWindFieldTest(parameterized.TestCase):

  def setUp(self):
    super(GridBasedWindFieldTest, self).setUp()
    self.key = jax.random.PRNGKey(0)
    self.x = units.Distance(m=0.0)
    self.y = units.Distance(m=0.0)
    self.pressure = 9000.0
    self.elapsed_time = dt.timedelta(seconds=0)

    self.wf = grid_based_wind_field.GridBasedWindField(SimpleWindFieldSampler())

  def test_grid_based_wind_field_returns_consistent_forecast(self):
    self.wf.reset(self.key, test_helpers.START_DATE_TIME)
    fc1 = self.wf.get_forecast(self.x, self.y, self.pressure, self.elapsed_time)
    fc2 = self.wf.get_forecast(self.x, self.y, self.pressure, self.elapsed_time)

    self.assertEqual(fc1, fc2)

  def test_grid_based_wind_field_returns_consistent_true_wind(self):
    self.wf.reset(self.key, test_helpers.START_DATE_TIME)
    wind1 = self.wf.get_ground_truth(
        self.x, self.y, self.pressure, self.elapsed_time)
    wind2 = self.wf.get_ground_truth(
        self.x, self.y, self.pressure, self.elapsed_time)

    self.assertEqual(wind1, wind2)

  def test_grid_based_wind_field_returns_different_forecast_and_true_wind(
      self):
    self.wf.reset(self.key, test_helpers.START_DATE_TIME)
    fc = self.wf.get_forecast(
        self.x, self.y, self.pressure, self.elapsed_time)
    wind = self.wf.get_ground_truth(
        self.x, self.y, self.pressure, self.elapsed_time)

    self.assertNotEqual(fc, wind)

  @parameterized.named_parameters(
      dict(
          testcase_name='x_direction',
          x1=units.Distance(km=-300.0),
          x2=units.Distance(km=-250.0),
          y1=units.Distance(km=0.0),
          y2=units.Distance(km=0.0),
          pressure1=9_000.0,
          pressure2=9_000.0,
          elapsed_time1=dt.timedelta(hours=0),
          elapsed_time2=dt.timedelta(hours=0)),
      dict(
          testcase_name='y_direction',
          x1=units.Distance(km=0.0),
          x2=units.Distance(km=0.0),
          y1=units.Distance(km=0.0),
          y2=units.Distance(km=50.0),
          pressure1=9_000.0,
          pressure2=9_000.0,
          elapsed_time1=dt.timedelta(hours=0),
          elapsed_time2=dt.timedelta(hours=0)),
      dict(
          testcase_name='pressure_direction',
          x1=units.Distance(km=0.0),
          x2=units.Distance(km=0.0),
          y1=units.Distance(km=0.0),
          y2=units.Distance(km=0.0),
          pressure1=9_000.0,
          pressure2=8_000.0,
          elapsed_time1=dt.timedelta(hours=0),
          elapsed_time2=dt.timedelta(hours=0)),
      dict(
          testcase_name='time_direction',
          x1=units.Distance(km=0.0),
          x2=units.Distance(km=0.0),
          y1=units.Distance(km=0.0),
          y2=units.Distance(km=0.0),
          pressure1=9_000.0,
          pressure2=9_000.0,
          elapsed_time1=dt.timedelta(hours=6),
          elapsed_time2=dt.timedelta(hours=12)),)
  def test_grid_based_wind_field_interpolates_correctly_between_grid_points(
      self,
      x1: units.Distance,
      x2: units.Distance,
      y1: units.Distance,
      y2: units.Distance,
      pressure1: float,
      pressure2: float,
      elapsed_time1: dt.timedelta,
      elapsed_time2: dt.timedelta):
    # For this test, x, y, pressure and elapsed time values have been
    # pre-calculated to lie on grid points.
    # Interpolation occurs for the forecast only.
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    mid_pressure = (pressure1 + pressure2) / 2.0
    mid_elapsed_time = (elapsed_time1 + elapsed_time2) / 2.0

    self.wf.reset(self.key, test_helpers.START_DATE_TIME)
    fc1 = self.wf.get_forecast(x1, y1, pressure1, elapsed_time1)
    fc2 = self.wf.get_forecast(x2, y2, pressure2, elapsed_time2)
    fc_mid = self.wf.get_forecast(mid_x, mid_y, mid_pressure, mid_elapsed_time)

    expected_vector = wind_field.WindVector(
        (fc1.u + fc2.u) / 2.0,
        (fc1.v + fc2.v) /2.0)
    self.assertAlmostEqual(fc_mid.u.meters_per_second,
                           expected_vector.u.meters_per_second,
                           places=5)
    self.assertAlmostEqual(fc_mid.v.meters_per_second,
                           expected_vector.v.meters_per_second,
                           places=5)

  def test_grid_based_wind_field_boomerangs_correctly(self):
    # By default, grid-based wind fields are 48 hours long.
    # These times should all be equivalent with a 48 hour boomerang.
    t1 = dt.timedelta(hours=46)
    t2 = dt.timedelta(hours=50)
    t3 = dt.timedelta(hours=46 + 48 * 2)
    t4 = dt.timedelta(hours=50 + 48 * 2)

    # Just to prove its boomeranging that makes these equal, here's a
    # time that should have different winds.
    non_equal_time = dt.timedelta(hours=49)

    self.wf.reset(self.key, test_helpers.START_DATE_TIME)
    fc1 = self.wf.get_forecast(self.x, self.y, self.pressure, t1)
    fc2 = self.wf.get_forecast(self.x, self.y, self.pressure, t2)
    fc3 = self.wf.get_forecast(self.x, self.y, self.pressure, t3)
    fc4 = self.wf.get_forecast(self.x, self.y, self.pressure, t4)
    non_equal_fc = self.wf.get_forecast(
        self.x, self.y, self.pressure, non_equal_time)

    self.assertEqual(fc1, fc2)
    self.assertEqual(fc1, fc3)
    self.assertEqual(fc1, fc4)
    self.assertNotEqual(fc1, non_equal_fc)

  @parameterized.named_parameters(
      dict(
          testcase_name='x_direction',
          x1=units.Distance(km=-500.0),
          x2=units.Distance(km=-550.0),
          y1=units.Distance(km=0.0),
          y2=units.Distance(km=0.0),
          pressure1=9_000.0,
          pressure2=9_000.0),
      dict(
          testcase_name='y_direction',
          x1=units.Distance(km=0.0),
          x2=units.Distance(km=0.0),
          y1=units.Distance(km=500.0),
          y2=units.Distance(km=900.0),
          pressure1=9_000.0,
          pressure2=9_000.0),
      dict(
          testcase_name='pressure_direction',
          x1=units.Distance(km=0.0),
          x2=units.Distance(km=0.0),
          y1=units.Distance(km=0.0),
          y2=units.Distance(km=0.0),
          pressure1=5_000.0,
          pressure2=1_000.0))
  def test_grid_based_wind_field_extends_wind_field_beyond_grid(
      self,
      x1: units.Distance,
      x2: units.Distance,
      y1: units.Distance,
      y2: units.Distance,
      pressure1: float,
      pressure2: float):
    # Note: all chosen parameters are at the edge of the grid or beyond it.
    self.wf.reset(self.key, test_helpers.START_DATE_TIME)
    fc1 = self.wf.get_forecast(x1, y1, pressure1, self.elapsed_time)
    fc2 = self.wf.get_forecast(x2, y2, pressure2, self.elapsed_time)

    self.assertEqual(fc1, fc2)

  def test_grid_based_wind_field_get_wind_column_matches_get_ground_truth(self):
    pressures = tuple(range(5_000, 15_000, 1_000))

    self.wf.reset(self.key, test_helpers.START_DATE_TIME)
    forecasts = [self.wf.get_forecast(self.x, self.y, p, self.elapsed_time)
                 for p in pressures]
    fc_column = self.wf.get_forecast_column(
        self.x, self.y, pressures, self.elapsed_time)

    self.assertEqual(forecasts, fc_column)


if __name__ == '__main__':
  absltest.main()
