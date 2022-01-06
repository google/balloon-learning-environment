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

"""Tests for envelope_safety."""

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import envelope_safety
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import test_helpers
import jax


class EnvelopeSafetyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='superpressure_low_critical_down',
          superpressure=50.0,
          input_action=control.AltitudeControlCommand.DOWN,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='superpressure_low_critical_stay',
          superpressure=50.0,
          input_action=control.AltitudeControlCommand.STAY,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='superpressure_low_critical_up',
          superpressure=50.0,
          input_action=control.AltitudeControlCommand.UP,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='superpressure_low_down',
          superpressure=200.0,
          input_action=control.AltitudeControlCommand.DOWN,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='superpressure_low_stay',
          superpressure=200.0,
          input_action=control.AltitudeControlCommand.STAY,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='superpressure_low_up',
          superpressure=200.0,
          input_action=control.AltitudeControlCommand.UP,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='superpressure_ok_down',
          superpressure=1000.0,
          input_action=control.AltitudeControlCommand.DOWN,
          expected_action=control.AltitudeControlCommand.DOWN),
      dict(
          testcase_name='superpressure_ok_stay',
          superpressure=1000.0,
          input_action=control.AltitudeControlCommand.STAY,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='superpressure_ok_up',
          superpressure=1000.0,
          input_action=control.AltitudeControlCommand.UP,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='superpressure_high_down',
          superpressure=2180.0,
          input_action=control.AltitudeControlCommand.DOWN,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='superpressure_high_stay',
          superpressure=2180.0,
          input_action=control.AltitudeControlCommand.STAY,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='superpressure_high_up',
          superpressure=2180.0,
          input_action=control.AltitudeControlCommand.UP,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='superpressure_high_critical_down',
          superpressure=2280.0,
          input_action=control.AltitudeControlCommand.DOWN,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='superpressure_high_critical_stay',
          superpressure=2280.0,
          input_action=control.AltitudeControlCommand.STAY,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='superpressure_high_critical_up',
          superpressure=2280.0,
          input_action=control.AltitudeControlCommand.UP,
          expected_action=control.AltitudeControlCommand.UP),
      )
  def test_envelope_safety_layer_alters_actions_correctly(
      self, superpressure: float, input_action: control.AltitudeControlCommand,
      expected_action: control.AltitudeControlCommand):
    atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))
    b = test_helpers.create_balloon(atmosphere=atmosphere)
    envelope_safety_layer = envelope_safety.EnvelopeSafetyLayer(
        b.state.envelope_max_superpressure)

    action = envelope_safety_layer.get_action(input_action, superpressure)

    self.assertEqual(action, expected_action)


if __name__ == '__main__':
  absltest.main()
