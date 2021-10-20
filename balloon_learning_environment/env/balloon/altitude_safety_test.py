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

"""Tests for altitude_safety."""

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env.balloon import altitude_safety
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import units
import jax


class AltitudeSafetyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))

    very_low_altitude = (
        altitude_safety.MIN_ALTITUDE - units.Distance(feet=100.0))
    low_altitude = (altitude_safety.MIN_ALTITUDE + altitude_safety.BUFFER / 2.0)
    low_nominal_altitude = (
        altitude_safety.MIN_ALTITUDE + altitude_safety.BUFFER +
        altitude_safety.RESTART_HYSTERESIS / 2.0)
    nominal_altitude = (
        altitude_safety.MIN_ALTITUDE + altitude_safety.BUFFER +
        altitude_safety.RESTART_HYSTERESIS + units.Distance(feet=100.0))

    self.very_low_altitude_pressure = self.atmosphere.at_height(
        very_low_altitude).pressure
    self.low_altitude_pressure = self.atmosphere.at_height(
        low_altitude).pressure
    self.low_nominal_altitude_pressure = self.atmosphere.at_height(
        low_nominal_altitude).pressure
    self.nominal_altitude_pressure = self.atmosphere.at_height(
        nominal_altitude).pressure

    self.pressures = {
        'very_low_altitude_pressure': self.very_low_altitude_pressure,
        'low_altitude_pressure': self.low_altitude_pressure,
        'low_nominal_altitude_pressure': self.low_nominal_altitude_pressure,
        'nominal_altitude_pressure': self.nominal_altitude_pressure
    }

  @parameterized.named_parameters(
      dict(
          testcase_name='very_low_atltitude_advises_up',
          pressure='very_low_altitude_pressure',
          action=control.AltitudeControlCommand.DOWN,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='low_altitude_advises_stay',
          pressure='low_altitude_pressure',
          action=control.AltitudeControlCommand.DOWN,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='nominal_altitude_allows_action',
          pressure='nominal_altitude_pressure',
          action=control.AltitudeControlCommand.DOWN,
          expected_action=control.AltitudeControlCommand.DOWN),
      dict(
          testcase_name='low_altitude_allows_up_action',
          pressure='low_altitude_pressure',
          action=control.AltitudeControlCommand.UP,
          expected_action=control.AltitudeControlCommand.UP))
  def test_safety_layer_gives_correct_action(
      self, pressure: str, action: control.AltitudeControlCommand,
      expected_action: control.AltitudeControlCommand):
    asl = altitude_safety.AltitudeSafetyLayer()
    pressure = self.pressures[pressure]

    action = asl.get_action(action, self.atmosphere, pressure)

    self.assertEqual(action, expected_action)

  @parameterized.named_parameters(
      dict(
          testcase_name='very_low_altitude_is_paused',
          pressure='very_low_altitude_pressure',
          expected=True),
      dict(
          testcase_name='low_altitude_is_paused',
          pressure='low_altitude_pressure',
          expected=True),
      dict(
          testcase_name='nominal_altitude_is_not_paused',
          pressure='nominal_altitude_pressure',
          expected=False))
  def test_navigation_is_paused_is_calculated_correctly(self, pressure: str,
                                                        expected: bool):
    asl = altitude_safety.AltitudeSafetyLayer()
    pressure = self.pressures[pressure]

    asl.get_action(control.AltitudeControlCommand.DOWN, self.atmosphere,
                   pressure)

    self.assertEqual(asl.navigation_is_paused, expected)

  def test_increasing_altitude_below_hysteresis_does_not_resume_control(self):
    asl = altitude_safety.AltitudeSafetyLayer()

    # Sets state to LOW.
    asl.get_action(control.AltitudeControlCommand.DOWN, self.atmosphere,
                   self.low_altitude_pressure)
    asl.get_action(control.AltitudeControlCommand.DOWN, self.atmosphere,
                   self.low_nominal_altitude_pressure)

    self.assertTrue(asl.navigation_is_paused)

  def test_increasing_altitude_above_hysteresis_resumes_control(self):
    asl = altitude_safety.AltitudeSafetyLayer()

    # Sets state to LOW.
    asl.get_action(control.AltitudeControlCommand.DOWN, self.atmosphere,
                   self.low_altitude_pressure)
    asl.get_action(control.AltitudeControlCommand.DOWN, self.atmosphere,
                   self.nominal_altitude_pressure)

    self.assertFalse(asl.navigation_is_paused)


if __name__ == '__main__':
  absltest.main()
