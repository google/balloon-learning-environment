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

"""Tests for balloon_learning_environment.env.balloon."""

import datetime as dt
import functools
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import test_helpers
from balloon_learning_environment.utils import units
import jax


class BalloonTest(parameterized.TestCase):

  def setUp(self):
    super(BalloonTest, self).setUp()
    self._wind_vector = wind_field.WindVector(
        units.Velocity(mps=3.0), units.Velocity(mps=-4.0))
    self.atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))
    self.create_balloon = functools.partial(
        test_helpers.create_balloon, atmosphere=self.atmosphere)

  def test_separate_state_from_functionality(self):
    """Tests that methods can't be called after extracting the state."""
    b = self.create_balloon()

    balloon_state = b.state
    self.assertIsInstance(balloon_state, balloon.BalloonState)
    self.assertNotIsInstance(balloon_state, balloon.Balloon)

  def test_balloon_lat_lng_is_correctly_calculated(self):
    # 111 kilometers is about 1 degree latitude and longitude at the equator.
    # This is a rough test - it is mostly a sanity check. The function
    # that calculates this is tested in spherical_geometry_test.py.
    b = self.create_balloon(
        x=units.Distance(km=111.0), y=units.Distance(km=111.0))

    self.assertAlmostEqual(b.state.latlng.lat().degrees, 1.0, places=1)
    self.assertAlmostEqual(b.state.latlng.lng().degrees, 1.0, places=1)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_excess_at_night',
          charge=1.0,
          hour=0,
          expected=False),
      dict(
          testcase_name='no_excess_at_day_low_power',
          charge=0.5,
          hour=12,
          expected=False),
      dict(
          testcase_name='excess_at_noon_high_power',
          charge=1.0,
          hour=12,
          expected=True))
  def test_excess_energy_calculated_correctly(
      self, charge: float, hour: int, expected: bool):
    b_state = self.create_balloon(
        power_percent=charge, date_time=units.datetime(2021, 9, 9, hour)).state

    self.assertEqual(b_state.excess_energy, expected)

  @parameterized.parameters((5235, 1234, 1.2357), (5235, -52, 1.0))
  def test_pressure_ratio_calculated_correctly(
      self, pressure, superpressure, expected_ratio):
    b_state = self.create_balloon().state
    b_state.pressure = pressure
    b_state.superpressure = superpressure
    self.assertAlmostEqual(b_state.pressure_ratio, expected_ratio, places=3)

  def test_balloon_goes_in_wind_direction(self):
    b = self.create_balloon()
    self.assertAlmostEqual(b.state.x.meters, 0)
    self.assertAlmostEqual(b.state.y.meters, 0)
    wv = wind_field.WindVector(
        units.Velocity(mps=10.0), units.Velocity(mps=12.0))

    b.simulate_step(wv, self.atmosphere, control.AltitudeControlCommand.STAY,
                    dt.timedelta(seconds=10.0))

    self.assertGreater(b.state.x.meters, 0)
    self.assertGreater(b.state.y.meters, 0)

  def test_balloon_goes_up_when_low(self):
    # use_stable_init=False means we don't update the mols air in the balloon
    # to corespond to the assigned ambient pressure.
    pressure0 = 20_123.0
    b = self.create_balloon(pressure=pressure0, use_stable_init=False)

    b.simulate_step(self._wind_vector, self.atmosphere,
                    control.AltitudeControlCommand.STAY,
                    dt.timedelta(seconds=10.0))

    self.assertLess(b.state.pressure, pressure0)

  def test_balloon_goes_down_when_high(self):
    # use_stable_init=False means we don't update the mols air in the balloon
    # to corespond to the assigned ambient pressure.
    pressure0 = 2345.0
    b = self.create_balloon(pressure=pressure0, use_stable_init=False)

    b.simulate_step(self._wind_vector, self.atmosphere,
                    control.AltitudeControlCommand.STAY,
                    dt.timedelta(seconds=10.0))

    self.assertGreater(b.state.pressure, pressure0)

  # TODO(scandido): We should add these tests once we have a function that cold
  # starts the sim in a stable state. Otherwise to disentangle dynamics not
  # related to the altitude control system from what we are trying to test we'd
  # need to run a mini-sim until the balloon stabilizes, and then run our test.
  # def test_balloon_goes_down_when_air_added(self):
  # def test_balloon_goes_up_when_air_vented(self):

  @unittest.mock.patch.object(
      solar, 'solar_calculator', return_value=(25.0, None, 1350))
  def test_balloon_charges_in_the_sun(self, _):
    b = self.create_balloon(power_percent=0.5)
    battery_soc0 = b.state.battery_soc
    wv = wind_field.WindVector(
        units.Velocity(mps=10.0), units.Velocity(mps=12.0))

    b.simulate_step(
        wv,
        self.atmosphere,
        control.AltitudeControlCommand.STAY,  # Not using ACS.
        dt.timedelta(seconds=10.0))

    self.assertGreater(b.state.battery_soc, battery_soc0)

  @unittest.mock.patch.object(
      solar, 'solar_calculator', return_value=(-25.0, None, 1350))
  def test_balloon_doesnt_charge_in_the_night(self, _):
    b = self.create_balloon(power_percent=0.5)
    battery_soc0 = b.state.battery_soc
    wv = wind_field.WindVector(
        units.Velocity(mps=10.0), units.Velocity(mps=12.0))

    b.simulate_step(
        wv,
        self.atmosphere,
        control.AltitudeControlCommand.STAY,  # Not using ACS.
        dt.timedelta(seconds=10.0))

    self.assertLess(b.state.battery_soc, battery_soc0)

  @unittest.mock.patch.object(
      solar, 'solar_calculator', return_value=(-25.0, None, 1350))
  def test_balloon_drains_hotel_load_at_night(self, _):
    b = self.create_balloon()
    wv = wind_field.WindVector(
        units.Velocity(mps=10.0), units.Velocity(mps=12.0))

    b.simulate_step(
        wv,
        self.atmosphere,
        control.AltitudeControlCommand.STAY,  # Not using ACS.
        dt.timedelta(seconds=10))

    self.assertEqual(b.state.power_load, b.state.nighttime_power_load)

  @unittest.mock.patch.object(
      solar, 'solar_calculator', return_value=(25.0, None, 1350))
  def test_balloon_drains_hotel_load_during_day(self, _):
    b = self.create_balloon()
    wv = wind_field.WindVector(
        units.Velocity(mps=10.0), units.Velocity(mps=12.0))

    b.simulate_step(
        wv,
        self.atmosphere,
        control.AltitudeControlCommand.STAY,  # Not using ACS.
        dt.timedelta(seconds=10.0))

    self.assertEqual(b.state.power_load, b.state.daytime_power_load)

  @unittest.mock.patch.object(
      solar, 'solar_calculator', return_value=(25.0, None, 1350))
  def test_acs_contributes_to_power_load(self, _):
    b = self.create_balloon()
    wv = wind_field.WindVector(
        units.Velocity(mps=10.0), units.Velocity(mps=12.0))

    b.simulate_step(
        wv,
        self.atmosphere,
        control.AltitudeControlCommand.DOWN,  # Using ACS.
        dt.timedelta(seconds=10.0))

    self.assertGreater(b.state.power_load, b.state.daytime_power_load)


if __name__ == '__main__':
  absltest.main()
