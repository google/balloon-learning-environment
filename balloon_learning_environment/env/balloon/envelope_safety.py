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

"""A safety layer that prevents balloons from bursting or zeropressuring."""

import enum
import logging

from balloon_learning_environment.env.balloon import control
import transitions

# There are two dominant failure modes - the balloon zero pressuring and
# losing lift, or bursting at maximum superpressure.
# To prevent these events, we:
#   1. Release air from the ballonet if zero pressure or bursting are
#     imminent - i.e. we are within CRITICAL_BUFFER.
#     If superpressure is critically low, we release air from the ballonet
#     (ascend), allowing the superpressure to increase by navigating to
#     altitudes that have lower pressure. If superpressure is critically
#     high, we also vent air to reduce the volume of gas in the envelope.
#     Either way, the action is ascend.
#   2. Prevent descending if we are heading toward zero pressure or burst -
#     i.e. we are within BUFFER.
#     This is because the balloon is at the edge of the effective altitude
#     range, so the safety layer prevents actions that would cause the
#     balloon from exceeding the effective range and breaking something.
# We use RESTART_HYSTERESIS to decide when to resume control.
CRITICAL_BUFFER = 150  # [Pa]
BUFFER = 250  # [Pa]
RESTART_HYSTERESIS = 50  # [Pa]


class _SuperpressureState(enum.Enum):
  NOMINAL = 0
  LOW_CRITICAL = 1
  LOW = 2
  HIGH = 3
  HIGH_CRITICAL = 4

# Note: Transitions are applied in the order of the first match.
# '*' is a catch-all, and applies to any state.
_ENVELOPE_SAFETY_TRANSITIONS = (
    dict(
        trigger='low_critical',
        source='*',
        dest=_SuperpressureState.LOW_CRITICAL),
    dict(
        trigger='low',
        source='*',
        dest=_SuperpressureState.LOW),
    dict(
        trigger='low_nominal',
        source=(_SuperpressureState.LOW_CRITICAL, _SuperpressureState.LOW),
        dest=_SuperpressureState.LOW),
    dict(
        trigger='low_nominal',
        source='*',
        dest=_SuperpressureState.NOMINAL),
    dict(
        trigger='nominal',
        source='*',
        dest=_SuperpressureState.NOMINAL),
    dict(
        trigger='high_nominal',
        source=(_SuperpressureState.HIGH, _SuperpressureState.HIGH_CRITICAL),
        dest=_SuperpressureState.HIGH),
    dict(
        trigger='high_nominal',
        source='*',
        dest=_SuperpressureState.NOMINAL),
    dict(
        trigger='high',
        source='*',
        dest=_SuperpressureState.HIGH),
    dict(
        trigger='high_critical', source='*',
        dest=_SuperpressureState.HIGH_CRITICAL),
)


class EnvelopeSafetyLayer:
  """A safety layer that protects the envelope from burst/low pressure.

  Attributes:
    navigation_is_paused: True if navigation is paused to reserve power.
  """

  def __init__(self, max_superpressure: float):
    """Constructor for envelope safety layer."""
    self._state_machine = transitions.Machine(
        states=_SuperpressureState,
        transitions=_ENVELOPE_SAFETY_TRANSITIONS,
        initial=_SuperpressureState.NOMINAL)
    logging.getLogger('transitions').setLevel(logging.WARNING)
    self._max_superpressure = max_superpressure

  def get_action(
      self, action: control.AltitudeControlCommand,
      superpressure: float) -> control.AltitudeControlCommand:
    """Gets an action recommended by the envelope safety layer.

    If the suprepressure is too high or too low, the safety layer will not
    allow the balloon to descend. If the superpressure is extremely high or
    extremely low, the balloon will release air from the ballonet (i.e. ascend).

    Args:
      action: The action that is attempting to be executed on the balloon.
      superpressure: The current superpressure of hte balloon.

    Returns:
      The action the balloon should execute.
    """
    self._transition_state(superpressure)

    if (self._state_machine.state in
        (_SuperpressureState.LOW_CRITICAL, _SuperpressureState.HIGH_CRITICAL)):
      return control.AltitudeControlCommand.UP

    if (self._state_machine.state in
        (_SuperpressureState.LOW, _SuperpressureState.HIGH)):
      if action == control.AltitudeControlCommand.DOWN:
        return control.AltitudeControlCommand.STAY

    return action

  @property
  def navigation_is_paused(self):
    return self._state_machine.state != _SuperpressureState.NOMINAL

  def _transition_state(self, superpressure: float):
    """Transitions the state of the safety layer."""
    if superpressure < CRITICAL_BUFFER:
      self._state_machine.low_critical()
    elif superpressure < BUFFER:
      self._state_machine.low()
    elif superpressure < BUFFER + RESTART_HYSTERESIS:
      self._state_machine.low_nominal()
    elif superpressure < self._max_superpressure - BUFFER - RESTART_HYSTERESIS:
      self._state_machine.nominal()
    elif superpressure < self._max_superpressure - BUFFER:
      self._state_machine.high_nominal()
    elif superpressure < self._max_superpressure - CRITICAL_BUFFER:
      self._state_machine.high()
    else:
      self._state_machine.high_critical()
