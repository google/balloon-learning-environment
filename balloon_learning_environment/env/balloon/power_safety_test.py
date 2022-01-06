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

"""Tests for power_safety."""

import datetime as dt

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import power_safety
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import test_helpers
from balloon_learning_environment.utils import units
import jax

import s2sphere as s2

_NIGHTTIME_HOTEL_LOAD = units.Power(watts=183.7)
_BATTERY_CAPACITY = units.Energy(watt_hours=2000.0)


class PowerSafetyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='night_low_power_prevents_action',
          date_time=units.datetime(2021, 6, 1, 0),
          battery_charge_percent=0.1,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='night_high_power_allows_action',
          date_time=units.datetime(2021, 6, 1, 0),
          battery_charge_percent=1.0,
          expected_action=control.AltitudeControlCommand.DOWN),
      dict(
          testcase_name='day_allows_action',
          date_time=units.datetime(2021, 6, 1, 12),
          battery_charge_percent=0.1,
          expected_action=control.AltitudeControlCommand.DOWN))
  def test_power_safety_layer_correctly_modifies_actions(
      self, date_time: dt.datetime, battery_charge_percent: float,
      expected_action: control.AltitudeControlCommand):
    # Initialize balloon at midnight.
    safety_layer = power_safety.PowerSafetyLayer(
        s2.LatLng.from_degrees(0.0, 0.0), date_time)

    action = safety_layer.get_action(control.AltitudeControlCommand.DOWN,
                                     date_time, _NIGHTTIME_HOTEL_LOAD,
                                     _BATTERY_CAPACITY * battery_charge_percent,
                                     _BATTERY_CAPACITY)

    self.assertEqual(action, expected_action)

  def test_power_safety_layer_correctly_forecasts_battery_charge(self):
    # We predict sunrise at 5:43 at our latlng and altitude.
    # Initialize at 0:43, exactly 5 hours before sunrise.
    date_time = units.datetime(2021, 8, 26, 0, 43)
    safety_layer1 = power_safety.PowerSafetyLayer(
        s2.LatLng.from_degrees(0.0, 0.0), date_time)
    safety_layer2 = power_safety.PowerSafetyLayer(
        s2.LatLng.from_degrees(0.0, 0.0), date_time)

    # Use round numbers to see when we will fall below 2.5% charge by sunrise.
    # After 5 hours 30 mins (sunrise + hysteresis) we will have
    # battery_charge - 5.5 watt_hours charge.
    action1 = safety_layer1.get_action(
        control.AltitudeControlCommand.DOWN,
        date_time,
        nighttime_power_load=units.Power(watts=1.0),
        battery_charge=units.Energy(watt_hours=7.9),
        battery_capacity=units.Energy(watt_hours=100.0))
    action2 = safety_layer2.get_action(
        control.AltitudeControlCommand.DOWN,
        date_time,
        nighttime_power_load=units.Power(watts=1.0),
        battery_charge=units.Energy(watt_hours=8.1),
        battery_capacity=units.Energy(watt_hours=100.0))

    self.assertEqual(action1, control.AltitudeControlCommand.STAY)
    self.assertEqual(action2, control.AltitudeControlCommand.DOWN)

  def test_power_safety_prevents_acting_on_low_power_at_night(self):
    # Create a balloon with 10% power at midnight.
    atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))
    b = test_helpers.create_balloon(
        power_percent=0.1,
        date_time=units.datetime(2020, 1, 1, 0, 0, 0),
        atmosphere=atmosphere)
    power_safety_layer = power_safety.PowerSafetyLayer(b.state.latlng,
                                                       b.state.date_time)

    for action in control.AltitudeControlCommand:
      with self.subTest(action.name):
        effective_action = power_safety_layer.get_action(
            action, b.state.date_time, b.state.nighttime_power_load,
            b.state.battery_charge, b.state.battery_capacity)

        # Safety layer only prevents balloons from going down.
        if action == control.AltitudeControlCommand.DOWN:
          expected_action = control.AltitudeControlCommand.STAY
        else:
          expected_action = action

        self.assertEqual(effective_action, expected_action)


if __name__ == '__main__':
  absltest.main()
