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

"""Tests for stable_params."""

import datetime as dt

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon import thermal
from balloon_learning_environment.utils import test_helpers
from balloon_learning_environment.utils import units
import jax


class StableParamsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(38))

  @parameterized.named_parameters(
      dict(testcase_name='middle_pressure', init_pressure=9_500.0),
      dict(testcase_name='high_pressure', init_pressure=11_500.0),
      dict(testcase_name='low_pressure', init_pressure=6_500.0))
  def test_cold_start_to_stable_params_initializes_mols_air_correctly(
      self, init_pressure: float):
    # The superpressure is very sensitive to temperature, and hence time of
    # day, so create the balloon at midnight.
    # create_balloon runs cold_start_to_stable_params by default.
    b = test_helpers.create_balloon(
        pressure=init_pressure,
        date_time=units.datetime(2020, 6, 1, 0, 0, 0),
        atmosphere=self.atmosphere)

    # Simulate the balloon for a while. If the internal parameters are
    # correctly set, it should stay at roughly the correct pressure level.
    for _ in range(100):
      b.simulate_step(
          wind_field.WindVector(
              units.Velocity(mps=3.0), units.Velocity(mps=-4.0)),
          self.atmosphere, control.AltitudeControlCommand.STAY,
          dt.timedelta(seconds=10.0))

    self.assertLess(abs(b.state.pressure - init_pressure), 100.0)

  @parameterized.named_parameters(
      dict(testcase_name='middle_pressure', init_pressure=9_500.0),
      dict(testcase_name='high_pressure', init_pressure=11_500.0),
      dict(testcase_name='low_pressure', init_pressure=5_000.0))
  def test_cold_start_to_stable_params_initializes_temperature_correctly(
      self, init_pressure: float):
    # create_balloon runs cold_start_to_stable_params by default.
    b = test_helpers.create_balloon(
        pressure=init_pressure, atmosphere=self.atmosphere)

    solar_elevation, _, solar_flux = solar.solar_calculator(
        b.state.latlng, b.state.date_time)
    d_internal_temp = thermal.d_balloon_temperature_dt(
        b.state.envelope_volume, b.state.envelope_mass,
        b.state.internal_temperature, b.state.ambient_temperature,
        b.state.pressure, solar_elevation, solar_flux,
        b.state.upwelling_infrared)

    # If the rate of change of temperature is low, the temperature is
    # initialized to a stable value.
    self.assertLess(d_internal_temp, 1e-3)


if __name__ == '__main__':
  absltest.main()
