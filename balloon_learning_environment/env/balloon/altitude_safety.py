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

"""Prevents the balloon navigating to unsafe altitudes.

This safety layer prevents the balloon from navigating to below 50,000 feet
of altitude. Internally, it maintains a state machine to remember whether
the balloon is close to the altitude limit. The balloon only returns to the
nominal state once the balloon has navigated sufficiently far from the
altitude limit. If the balloon moves below the altitude limit, the safety
layer will issue the ascend command.
"""

import enum
import logging

from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import units
import transitions

# TODO(joshgreaves): This may require some tuning.
BUFFER = units.Distance(feet=500.0)
RESTART_HYSTERESIS = units.Distance(feet=500.0)
MIN_ALTITUDE = units.Distance(feet=50_000.0)


class _AltitudeState(enum.Enum):
  NOMINAL = 0
  LOW = 1
  VERY_LOW = 2


# Note: Transitions are applied in the order of the first match.
# '*' is a catch-all, and applies to any state.
_ALTITUDE_SAFETY_TRANSITIONS = (
    dict(trigger='very_low', source='*', dest=_AltitudeState.VERY_LOW),
    dict(trigger='low', source='*', dest=_AltitudeState.LOW),
    dict(
        trigger='low_nominal',
        source=(_AltitudeState.VERY_LOW, _AltitudeState.LOW),
        dest=_AltitudeState.LOW),
    dict(
        trigger='low_nominal',
        source=_AltitudeState.NOMINAL,
        dest=_AltitudeState.NOMINAL),
    dict(trigger='nominal', source='*', dest=_AltitudeState.NOMINAL),
)


class AltitudeSafetyLayer:
  """A safety layer that prevents balloons navigating to unsafe altitudes."""

  def __init__(self):
    self._state_machine = transitions.Machine(
        states=_AltitudeState,
        transitions=_ALTITUDE_SAFETY_TRANSITIONS,
        initial=_AltitudeState.NOMINAL)
    logging.getLogger('transitions').setLevel(logging.WARNING)

  def get_action(self, action: control.AltitudeControlCommand,
                 atmosphere: standard_atmosphere.Atmosphere,
                 pressure: float) -> control.AltitudeControlCommand:
    """Gets the action recommended by the safety layer.

    Args:
      action: The action the controller has supplied to the balloon.
      atmosphere: The atmospheric conditions the balloon is flying in.
      pressure: The current pressure of the balloon.

    Returns:
      An action the safety layer recommends.
    """
    altitude = atmosphere.at_pressure(pressure).height
    self._transition_state(altitude)

    if self._state_machine.state == _AltitudeState.VERY_LOW:
      # If the balloon is too low, make it ascend.
      return control.AltitudeControlCommand.UP
    elif self._state_machine.state == _AltitudeState.LOW:
      # If the balloon is almost too low, don't let it go lower.
      if action == control.AltitudeControlCommand.DOWN:
        return control.AltitudeControlCommand.STAY

    return action

  @property
  def navigation_is_paused(self):
    return self._state_machine.state != _AltitudeState.NOMINAL

  def _transition_state(self, altitude: units.Distance):
    if altitude < MIN_ALTITUDE:
      self._state_machine.very_low()
    elif altitude < MIN_ALTITUDE + BUFFER:
      self._state_machine.low()
    elif altitude < MIN_ALTITUDE + BUFFER + RESTART_HYSTERESIS:
      self._state_machine.low_nominal()
    else:
      self._state_machine.nominal()
