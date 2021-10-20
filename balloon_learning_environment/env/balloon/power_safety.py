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


# TODO(joshgreaves): Why not allow ascend when navigation is paused?
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
    while date_time > self._sunrise:
      self._sunrise += dt.timedelta(days=1)
    while date_time > self._sunset:
      self._sunset += dt.timedelta(days=1)

    if self._sunset < self._sunrise:
      # It is daytime 🌞. For the system we are modeling, we don't need to
      # worry about power until night falls. However, this may be an issue
      # if the flight system changes or balloons are being flown at
      # latitudes/dates where days are very short.
      self.navigation_is_paused = False
      return action

    # Everything after here is nighttime 🌝.
    if self.navigation_is_paused:  # We've already decided to pause control.
      return control.AltitudeControlCommand.STAY

    # Decide whether we should pause control now.
    night_power = nighttime_power_load
    time_to_sunrise = self._sunrise - date_time
    floating_charge = night_power * time_to_sunrise

    # TODO(joshgreaves): Maybe use a function of nighttime hotel load
    # instead of a percentage of capacity.
    expected_remaining_normalized_charge = (battery_charge -
                                            floating_charge) / battery_capacity
    if expected_remaining_normalized_charge < 0.05:
      self.navigation_is_paused = True
      return control.AltitudeControlCommand.STAY

    # It's nighttime, but we aren't in danger of running out of power.
    return action
