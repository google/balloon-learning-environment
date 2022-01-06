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

"""An implementation of the StationSeeker controller.

The StationSeeker controller takes actions by means of a parametrized score
function that ranks pressure levels (altitudes). The controller then navigates
to the highest-scoring pressure level.

StationSeeker is discussed in detail in Bellemare, Candido, Castro, Gong,
Machado, Moitra, Ponda, and Wang (2020). It is a strong baseline controller
whose parameters were tuned from extensive simulation and real flight tests.

The parameters used here are those reported in the paper.
"""

from typing import Sequence, Tuple

from balloon_learning_environment.agents import agent
from balloon_learning_environment.env import features
from balloon_learning_environment.utils import transforms
import numpy as np


class StationSeekerAgent(agent.Agent):
  """Implementation of the StationSeeker controller."""

  def __init__(self, num_actions: int, observation_shape: Sequence[int]):
    del num_actions
    del observation_shape

    # StationSeeker constants. In comments: the equivalent labels in the paper.
    self.half_radius = 35  # used to compute alpha_Delta
    self.magnitude_weight = 0.07  # k_1
    self.close_bearing_weight = 0.6  # used to compute w_Delta
    self.far_bearing_weight = 0.45
    self.close_bearing = 250
    self.far_bearing = 500
    self.default_score = 0.5  # g_unknown
    self.hysteresis_k2 = 0.05  # k2
    self.hysteresis_k3 = 0.001  # k3
    self.confidence_epsilon = 0.01  # stabilizes control? TODO(bellemare): check

    self.max_altitude_score = 1 + self.hysteresis_k2 + self.confidence_epsilon

  def begin_episode(self, observation: np.ndarray) -> int:
    assert observation is not None

    return self.pick_action(observation)

  def step(self, reward: float, observation: np.ndarray) -> int:
    del reward
    assert observation is not None

    return self.pick_action(observation)

  def end_episode(self, reward: float, terminal: bool) -> None:
    pass

  def pick_action(self, features_as_vector: np.ndarray) -> int:
    """Picks the action based on the best pressure level."""
    named_features = features.NamedPerciatelliFeatures(features_as_vector)

    level, _ = self.find_best_pressure_level(named_features)
    midpoint = named_features.wind_column_center()

    # TODO(bellemare): Revert back to control macros, these currently do not
    # work.
    if level < midpoint:
      return 2  # control.AltitudeControlCommand.UP
    elif level > midpoint:
      return 0  # control.AltitudeControlCommand.DOWN
    else:
      return 1  # control.AltitudeControlCommand.STAY

  def find_best_pressure_level(
      self,
      named_features: features.NamedPerciatelliFeatures
      ) -> Tuple[int, np.ndarray]:
    """Compute the altitude score for each pressure level."""
    best_pressure_level = None
    best_score = 0

    scores = np.zeros(named_features.num_pressure_levels)

    # Score each pressure level in turn.
    for l in range(named_features.num_pressure_levels):
      if not named_features.level_is_valid(l):
        continue

      altitude_score = self.altitude_score(named_features, l)
      assert 0 <= altitude_score <= self.max_altitude_score

      scores[l] = altitude_score
      if altitude_score > best_score:
        best_score = altitude_score
        best_pressure_level = l

    # At least one pressure level should be valid.
    assert best_pressure_level is not None
    return best_pressure_level, scores

  def altitude_score(
      self,
      named_features: features.NamedPerciatelliFeatures,
      level: int) -> float:
    """Computes the score at the given pressure level.

    This combines the wind score with a hysteresis term plus the default score
    for uncertainty winds.

    Args:
      named_features: the features (with names).
      level: the queried pressure level.

    Returns:
      The score at the current level (between 0 and 2).
    """
    wind = named_features.wind_at(level)
    wind_score = self.wind_score(named_features, level)
    uncertainty = wind.uncertainty

    midpoint = named_features.wind_column_center()
    level_distance = np.abs(level - midpoint)

    # Extra cost for moving.
    hysteresis_term = (
        self.hysteresis_k2 * np.exp(-self.hysteresis_k3 * level_distance))

    return (
        (1.0 - uncertainty + self.confidence_epsilon) * wind_score +
        uncertainty * self.default_score +
        hysteresis_term)

  def wind_score(
      self,
      named_features: features.NamedPerciatelliFeatures,
      level: int) -> float:
    """Scores the wind magnitude and bearing at the given level."""
    # NOTE(bellemare): Currently this code assumes that named_features contains
    # normalized variables (mostly in [0, 1]). As a consequence, there is some
    # code (right below this comment) that first un-normalizes. If
    # named_features changes, this needs to be updated also.
    wind = named_features.wind_at(level)
    # De-normalize features. This is somewhat roundabout; needs a better
    # solution.
    wind = features.convert_wind_feature_to_real_wind(wind)

    distance = transforms.undo_squash_to_unit_interval(
        named_features.distance_to_station, 250.0)

    # Bearing cost varies with distance.
    bearing_weight_ramp = self.far_bearing_weight - self.close_bearing_weight
    bearing_weight_rate = self.far_bearing - self.close_bearing
    bearing_weight_coeff = np.clip(
        (distance - self.close_bearing) / bearing_weight_rate,
        0.0, 1.0)
    bearing_weight = (
        self.close_bearing_weight + bearing_weight_coeff * bearing_weight_ramp)

    alpha_delta = np.exp(-distance / self.half_radius)

    magnitude_term = np.exp(-self.magnitude_weight * wind.magnitude)
    bearing_term = np.exp(-bearing_weight * wind.bearing)

    return (1 - alpha_delta) * bearing_term + alpha_delta * magnitude_term
