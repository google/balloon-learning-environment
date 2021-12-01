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

"""Power safety layer to prevent balloon from running out of power."""
import datetime as dt

from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.utils import units

import s2sphere as s2


class PowerSafetyLayer():
  """A safety layer that prevents balloons from running out of power.

  Attributes:
    navigation_is_paused: True if navigation is paused to reserve power.
  """

  def __init__(self, latlng: s2.LatLng, date_time: dt.datetime):
    """Constructor for power safety layer.

    Args:
      latlng: The latitude and longitude of the balloon.
      date_time: The datetime of the balloon.
    """
    self._sunrise, self._sunset = solar.get_next_sunrise_sunset(
        latlng, date_time)
    self.navigation_is_paused = False

    self._soc_min = 0.025  # Don't let the battery SOC fall below this.
    self._soc_restart = 0.05  # Resume control when SOC is more than this.
    self._time_hysteresis = dt.timedelta(minutes=30)
    # Rather than timing everything to sunrise, we time it to 30 minutes
    # after sunrise, to ensure that the sun ☀️ has risen enough to shine on
    # the solar panels.
    self._sunrise_with_hysteresis = self._sunrise + self._time_hysteresis

  def get_action(
      self,
      action: control.AltitudeControlCommand,
      date_time: dt.datetime,
      nighttime_power_load: units.Power,
      battery_charge: units.Energy,
      battery_capacity: units.Energy,
  ) -> control.AltitudeControlCommand:
    """Gets an action recommended by the power safety layer.

    If the balloon has plenty of energy, the action will pass through.
    However, if the balloon is in danger of running out of energy, it will
    recommend a conservative action - stay.

    Args:
      action: The action that is attempting to be executed on the balloon.
      date_time: The date and time of the balloon.
      nighttime_power_load: The power the balloon consumes at nighttime
        (not including ACS).
      battery_charge: The current charge of the battery.
      battery_capacity: The capacity of the battery.

    Returns:
      The action the balloon should execute.
    """
    # Note: Sunrise/Sunset time can change if the balloon position changes.
    # We currently don't account for it, may want to revisit the decision
    # if this approximation is not adequate.
    # We also update the sunrise/sunset time naively, since it is a reasonable
    # approximation within a few minutes for episode lengths that are only
    # several days long.
    while date_time > self._sunrise_with_hysteresis:
      self._sunrise_with_hysteresis += dt.timedelta(days=1)
    while date_time > self._sunset:
      self._sunset += dt.timedelta(days=1)

    if self._sunset < self._sunrise_with_hysteresis:
      # If it is daytime but the battery charge is still very low, do not
      # resume control until the battery has more charge.
      soc = battery_charge / battery_capacity
      if self.navigation_is_paused and soc < self._soc_restart:
        return self.get_paused_action(action)

      # It is daytime 🌞. For the system we are modeling, we don't need to
      # worry about power until night falls. However, this may be an issue
      # if the flight system changes or balloons are being flown at
      # latitudes/dates where days are very short.
      self.navigation_is_paused = False
      return action

    # Everything after here is nighttime 🌝.
    if self.navigation_is_paused:  # We've already decided to pause control.
      return self.get_paused_action(action)

    # Decide whether we should pause control now.
    night_power = nighttime_power_load
    time_to_sunrise = self._sunrise_with_hysteresis - date_time
    floating_charge = night_power * time_to_sunrise

    expected_remaining_normalized_charge = (
        (battery_charge - floating_charge) / battery_capacity)
    if expected_remaining_normalized_charge < self._soc_min:
      self.navigation_is_paused = True
      return self.get_paused_action(action)

    # It's nighttime, but we aren't in danger of running out of power.
    return action

  @staticmethod
  def get_paused_action(
      action: control.AltitudeControlCommand) -> control.AltitudeControlCommand:
    # Down uses more power than up or stay, so we cannot allow it.
    if action == control.AltitudeControlCommand.DOWN:
      return control.AltitudeControlCommand.STAY
    return action
