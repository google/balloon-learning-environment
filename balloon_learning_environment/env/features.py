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

"""Creates a feature vector from the current simulator state.

This combines wind information from the WindGP with other variables of
interest.

TODO(bellemare): Discuss how the feature vector is constructed once the code
is completely in (Also cite Nature paper).

TODO(bellemare): Note to self (to be added to the above):
  levels are encoded as (uncertainty, bearing, magnitude)
  uncertainty ranges from 0 to 1 -- 0 is maximally confident, no uncertainty.

  Invalid levels are (0, 1, 1).
"""

import abc
import dataclasses
import datetime as dt
import math

from absl import logging
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env import wind_gp
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import power_table
from balloon_learning_environment.env.balloon import pressure_range_builder
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import transforms
from balloon_learning_environment.utils import units
import gin
import gym
import numpy as np

TOLERANCE = 1e-5


def compute_solar_angle(balloon_state: balloon.BalloonState) -> float:
  """Computes the solar angle relative to the balloon's position.

  Args:
    balloon_state: current state of the balloon.

  Returns:
    Solar angle at the balloon's position.
  """
  el_degree, _, _ = solar.solar_calculator(
      balloon_state.latlng,
      balloon_state.date_time)

  return el_degree


def compute_sunrise_time(balloon_state: balloon.BalloonState) -> float:
  """Return Sal Candido's normalized solar cycle time.

  This maps the current time into the [0, 2pi] interval, according to:

    [sunrise, sunset] -> [0, pi]
    [sunset, sunrise] -> [pi, 2pi]

  2pi is equivalent to 0 in sunrise time.

  Args:
    balloon_state: current state of the balloon.

  Returns:
    Time since last sunrise (in [0, 2pi]).
  """
  current_time = balloon_state.date_time
  sunrise, sunset = solar.get_next_sunrise_sunset(
      balloon_state.latlng, current_time)

  # Make sure get_next_sunrise_sunset returns times in the future.
  assert sunrise - dt.timedelta(days=1) <= current_time <= sunrise
  assert sunset - dt.timedelta(days=1) <= current_time <= sunset

  if sunset < sunrise:  # It's day time, sunset is up next.
    sunrise = sunrise - dt.timedelta(days=1)  # Get previous sunrise.
    # Proportion of time from sunrise to sunset.
    return math.pi * (current_time - sunrise) / (sunset - sunrise)
  else:  # It's night, sunrise is up next.
    sunset = sunset - dt.timedelta(days=1)
    # Proportion of time from sunset to sunrise.
    return math.pi + math.pi * (current_time - sunset) / (sunrise - sunset)


class FeatureConstructor(abc.ABC):
  """An interface for constructing features from the Balloon Arena.

  A feature constructor takes a forecast and multiple observations and
  constructs a feature vector as a numpy array.

  This interface requires an observe function since since the
  feature construction may require state tracking e.g. when using a
  Gaussian Process over the observed winds.
  """

  # TODO(joshgreaves): Is it ok to pass the atmosphere here?
  # This is the atmosphere we are flying with, so should we have a noisy
  # observation instead?
  @abc.abstractmethod
  def __init__(self,
               forecast: wind_field.WindField,
               atmosphere: standard_atmosphere.Atmosphere) -> None:
    """The FeatureConstructor constructor.

    The constructor of each child class should follow this function signature.

    Args:
      forecast: A forecast for the current arena.
      atmosphere: The current atmospheric conditions.
    """

  @abc.abstractmethod
  def observe(self, observation: simulator_data.SimulatorObservation) -> None:
    """Observes all sensor readings at the next timestep from the simulator."""

  @abc.abstractmethod
  def get_features(self) -> np.ndarray:
    """Gets the current feature vector given all observations."""

  @abc.abstractproperty
  def observation_space(self) -> gym.Space:
    """Gets the observation space specification for the feature vector."""


@dataclasses.dataclass
class PerciatelliWindFeature:
  """Encodes the triple of characteristics for the wind at a given pressure."""
  uncertainty: float
  bearing: float  # TODO(joshgreaves): Rename - angle error?
  magnitude: float

  def is_valid_wind(self):
    # A wind is valid as long as one of the three values is not extreme.
    # In theory, magnitude should suffice since a magnitude of 1.0 corresponds
    # to an infinite wind.
    return (self.magnitude != 1.0 or
            self.bearing != 1.0 or
            self.uncertainty != 0.0)


class NamedPerciatelliFeatures:
  """A helper class for parsing Perciatelli feature vectors.

  The class takes a numpy array of Perciatelli features
  (see PerciatelliFeatureConstructor) and assigns names to the elements
  of the feature vector. A large portion of the feature vector is the wind
  column, which can be accessed with the wind_at function.
  """

  def __init__(self, features: np.ndarray):
    # Some tests might try to pass in wrongly-sized observations. We make sure
    # that any code using this method has the right number of features.
    assert len(features) == 1099

    self._winds = features[16:]
    assert len(self._winds) % 3 == 0, 'Unexpected number of wind features.'
    self.num_pressure_levels = len(self._winds) // 3

    ambient_features = features[:16]
    # TODO(joshgreaves): Convert these into the right data types/units.
    #   Specifically, implement the reverse map for all of these. This also
    #   applies to the wind column data.
    # NOTE(bellemare): If these values are changed to reflect "raw" variables,
    # for example distance in kms versus normalized in [0, 1], then also
    # update station_seeker_agent.py.

    # Note(joshgreaves): pressure actually uses linear_rescale_with_saturation,
    # but the inverse function is equivalent. There may be some information
    # lost from the saturation.
    self.balloon_pressure = transforms.undo_linear_rescale_with_extrapolation(
        ambient_features[0],
        constants.PERCIATELLI_PRESSURE_RANGE_MIN,
        constants.PERCIATELLI_PRESSURE_RANGE_MAX)
    self.battery_charge = ambient_features[1]

    self.solar_elevation = ambient_features[2]
    self.sin_normalized_solar_cycle = ambient_features[3]
    self.cos_normalized_solar_cycle = ambient_features[4]

    self.sin_heading_to_station = ambient_features[5]
    self.cos_heading_to_station = ambient_features[6]
    self.distance_to_station = ambient_features[7]

    # Note the command order in the feature vector order is: up, stay, down.
    # Elsewhere it is: down, stay, up. This is due to historical reasons
    # and backwards compatability.
    last_command_idx = int(np.argmax(ambient_features[8:11]))
    if last_command_idx == 0:
      self.last_command = control.AltitudeControlCommand.UP
    elif last_command_idx == 1:
      self.last_command = control.AltitudeControlCommand.STAY
    elif last_command_idx == 2:
      self.last_command = control.AltitudeControlCommand.DOWN

    self.navigation_enabled = ambient_features[11]
    self.navigation_paused = ambient_features[12]

    self.has_excess_energy = ambient_features[13]
    self.descent_cost = ambient_features[14]
    self.internal_pressure_ratio = ambient_features[15]

  def wind_at(self, level: int) -> PerciatelliWindFeature:
    """Returns the (magnitude, bearing, uncertainty) triple at the given level.

    This level may not be valid, since it is looking at the centered wind
    column. Use `level_is_valid` to determine whether this is a valid
    (reachable) pressure level.

    Args:
      level: the query pressure level.
    """
    if 0 > level >= self.num_pressure_levels:
      raise ValueError(f'Invalid wind level: {level}')

    wind = self._winds[level * 3:level * 3 + 3]
    return PerciatelliWindFeature(*wind)

  def level_is_valid(self, level: int) -> bool:
    """Returns whether this is a valid pressure level."""
    return self.wind_at(level).is_valid_wind()

  def magnitude(self, level: int) -> float:
    return self.wind_at(level).magnitude

  def bearing(self, level: int) -> float:
    return self.wind_at(level).bearing

  def uncertainty(self, level: int) -> float:
    return self.wind_at(level).uncertainty

  def wind_column_center(self) -> int:
    """Returns the index of the center of the wind column."""
    assert self.num_pressure_levels % 2 == 1
    return self.num_pressure_levels // 2


# TODO(joshgreaves): Figure out where this goes.
def convert_wind_feature_to_real_wind(
    wind: PerciatelliWindFeature) -> PerciatelliWindFeature:
  return PerciatelliWindFeature(
      wind.uncertainty,
      transforms.undo_linear_rescale_with_extrapolation(
          wind.bearing, 0.0, math.pi),
      transforms.undo_squash_to_unit_interval(wind.magnitude, 30.0))


@gin.configurable(allowlist=[])
class PerciatelliFeatureConstructor(FeatureConstructor):
  """A feature constructor for Perciatelli features."""

  def __init__(self, forecast: wind_field.WindField,
               atmosphere: standard_atmosphere.Atmosphere) -> None:
    """Creates a new feature constructor object for Perciatelli features.

    Args:
      forecast: A forecast for the current arena.
      atmosphere: The atmospheric conditions the balloon is flying in.
    """
    super(PerciatelliFeatureConstructor, self).__init__(forecast, atmosphere)

    self.num_pressure_levels = 181
    self.min_pressure = constants.PERCIATELLI_PRESSURE_RANGE_MIN
    self.max_pressure = constants.PERCIATELLI_PRESSURE_RANGE_MAX

    # Discretize pressure into uniformly-spaced levels.
    self.pressure_levels = np.linspace(
        self.min_pressure, self.max_pressure, self.num_pressure_levels)

    # 3 features per pressure level in the encoding, which there are twice as
    # many of as we use a relative encoding. Plus sixteen ambient variables.
    self.num_features = 3 * (self.num_pressure_levels * 2 - 1) + 16

    self.windgp = wind_gp.WindGP(forecast)
    self._atmosphere = atmosphere
    self._last_balloon_state = None

  def observe(self, observation: simulator_data.SimulatorObservation) -> None:
    """Observes the latest observation and updates the internal state."""
    self._last_balloon_state = observation.balloon_observation
    self.windgp.observe(observation.balloon_observation.x,
                        observation.balloon_observation.y,
                        observation.balloon_observation.pressure,
                        observation.balloon_observation.time_elapsed,
                        observation.wind_at_balloon)

  def get_features(self) -> np.ndarray:
    """Returns the feature vector for the current balloon.

    Returns:
      feature_vector: a feature vector for the wind column + more.

    Raises:
      ValueError: if balloon_pressure is outside of the allowed range. This can
        be tested for with is_valid_pressure().
    """
    balloon_pressure = self._last_balloon_state.pressure
    if (balloon_pressure < self.min_pressure or
        balloon_pressure > self.max_pressure):
      logging.warning((
          'Balloon pressure %.2f not fully represented by feature constructor. '
          'If this happens frequently, consider changing the min/max pressure '
          'bounds on PerciatelliFeatureConstructor.'), balloon_pressure)

    feature_vector = np.zeros(self.num_features, dtype=np.float32)
    self._add_ambient_features(feature_vector)
    self._add_wind_features(feature_vector)

    return feature_vector

  @property
  def observation_space(self) -> gym.spaces.Box:
    """Returns the observation space for this feature constructor."""
    # Most features are in [0, 1].
    low = np.zeros(self.num_features, dtype=np.float32)
    high = np.ones(self.num_features, dtype=np.float32)

    # The following features use sine or cosine, so are in [-1, 1]
    trig_features = [3, 4, 5, 6]
    low[trig_features] = -1.0

    # The ACS pressure ratio feature is (pressure + superpressure) / pressure.
    # Therefore it is in [1, inf].
    low[15] = 1.0
    high[15] = np.inf

    return gym.spaces.Box(low=low, high=high)

  # NOTE(scandido): This should probably be renamed. Other pressures are very
  # much "valid", but our choice of feature vector would probably be poor for
  # balloons regularly flying in these ranges.
  def is_valid_pressure(self, pressure: float) -> bool:
    """Returns whether a given pressure in within the expected range."""
    return pressure >= self.min_pressure and pressure <= self.max_pressure

  def _nearest_pressure_level(
      self, pressure: float) -> int:
    """Returns the pressure level nearest to a given pressure value.

    Args:
      pressure: Desired pressure.

    Returns:
      level: The corresponding nearest level.
    """
    if pressure < self.min_pressure or pressure > self.max_pressure:
      # A warning has been logged. Quantize the pressure level.
      pressure = min(max(pressure, self.min_pressure), self.max_pressure)

    # Basically quantize 'pressure'.
    # Note: this assumes uniform pressure levels.
    assert len(self.pressure_levels) >= 2
    delta = self.pressure_levels[1] - self.pressure_levels[0]

    rescaled = (pressure - self.min_pressure) / delta
    level = int(round(rescaled))

    assert level >= 0 and level < self.num_pressure_levels
    return level

  def _add_ambient_features(self, feature_vector: np.ndarray) -> None:
    """Adds all ambient features to the feature vector.

    The ambient features are the first 16 features of the feature vector.

    Args:
      feature_vector: The feature vector to add the ambient features to. They
        will be inserted into feature_vector[:16].
    """
    balloon_state = self._last_balloon_state

    # 0: Pressure.
    feature_vector[0] = transforms.linear_rescale_with_saturation(
        balloon_state.pressure, self.min_pressure, self.max_pressure)

    # 1: Battery charge.
    feature_vector[1] = balloon_state.battery_soc

    # 2-4: Solar features.
    solar_angle = compute_solar_angle(self._last_balloon_state)
    feature_vector[2] = transforms.linear_rescale_with_saturation(
        solar_angle, -90.0, 90.0)

    sunrise_time = compute_sunrise_time(balloon_state)
    assert 0 <= sunrise_time <= 2 * math.pi + 1e-6
    # Sin of normalized solar cycle.
    feature_vector[3] = math.sin(sunrise_time)
    # Cos of normalized solar cycle.
    feature_vector[4] = math.cos(sunrise_time)

    # 5-7: Heading and distance to station.
    distance_to_station = units.relative_distance(
        balloon_state.x, balloon_state.y)
    # Heading is from North, and increases to the East.
    angle_heading_to_station = math.atan2(
        -balloon_state.x.kilometers, -balloon_state.y.kilometers)
    # Sin of heading to station.
    feature_vector[5] = math.sin(angle_heading_to_station)
    # Cos of heading to station.
    feature_vector[6] = math.cos(angle_heading_to_station)
    # Distance to station, km (normalized with x/(x+250)).
    feature_vector[7] = transforms.squash_to_unit_interval(
        distance_to_station.kilometers, 250)

    # 8-10: Last command.
    # Last command was ascend.
    feature_vector[8] = float(
        balloon_state.last_command == control.AltitudeControlCommand.UP)
    # Last command was stay.
    feature_vector[9] = float(
        balloon_state.last_command == control.AltitudeControlCommand.STAY)
    # Last command was descend.
    feature_vector[10] = float(
        balloon_state.last_command == control.AltitudeControlCommand.DOWN)

    # 11-12: Navigation is paused.
    # Navigation is paused.
    feature_vector[11] = float(
        self._last_balloon_state.navigation_is_paused)
    # Navigation is not paused.
    feature_vector[12] = float(
        not self._last_balloon_state.navigation_is_paused)

    # 13: Excess energy available.
    feature_vector[13] = float(self._last_balloon_state.excess_energy)
    # 14: ACS power to use.
    power_to_use = power_table.lookup(balloon_state.pressure_ratio,
                                      balloon_state.battery_soc)
    power_to_use = transforms.linear_rescale_with_saturation(power_to_use,
                                                             100, 300)
    feature_vector[14] = power_to_use

    # 15: Internal pressure ratio.
    feature_vector[15] = balloon_state.pressure_ratio

  def _add_wind_features(self, feature_vector: np.ndarray) -> None:
    """Adds all wind features to the feature vector.

    The wind features are found after the 16 ambient features. They are
    organized by increasing pressure level, with 3 features per pressure
    level. These features are (uncertainty, bearing, magnitude).

    Args:
      feature_vector: The feature vector to add the wind features to. They
        will be inserted into feature_vector[16:].
    """
    feature_index = 16  # The end of the ambient features.

    # Batch query the WindGP for the winds in the wind column.
    batch_query = np.zeros((self.num_pressure_levels, 4))
    batch_query[:, 0] = self._last_balloon_state.x.meters
    batch_query[:, 1] = self._last_balloon_state.y.meters
    batch_query[:, 2] = self.pressure_levels
    batch_query[:, 3] = self._last_balloon_state.time_elapsed.total_seconds()

    means, deviations = self.windgp.query_batch(batch_query)

    balloon_level = self._nearest_pressure_level(
        self._last_balloon_state.pressure)
    # Based on above clipping, balloon_level should be in range 0 ... 180.
    assert 0 <= balloon_level < self.num_pressure_levels

    num_encoded_levels = self.num_pressure_levels * 2 - 1

    # Determine the offset x such that x + b = 181, where b is the balloon's
    # level. This centers the balloon in the column.
    num_levels_lower = self.num_pressure_levels - balloon_level - 1
    num_levels_higher = (
        num_encoded_levels - num_levels_lower - self.num_pressure_levels)

    assert num_levels_higher >= 0

    # Pad the vector with unreachable data above the valid range (these
    # are lower-valued pressures).
    feature_index = self._add_unreachable_pressure_levels(
        feature_vector, feature_index, num_levels_lower)

    # Compute where the station is relative to us.
    # Since the station is at (0, 0), this is pretty easy.
    station_direction = -np.array(
        [self._last_balloon_state.x.meters, self._last_balloon_state.y.m])
    distance_to_station = units.relative_distance(self._last_balloon_state.x,
                                                  self._last_balloon_state.y)
    # Normalize for downstream use.
    station_direction /= (distance_to_station.meters + TOLERANCE)

    pressure_range = pressure_range_builder.get_pressure_range(
        self._last_balloon_state, self._atmosphere)

    # Add the wind data within the valid range.
    for level, pressure in enumerate(self.pressure_levels):

      # Check whether the pressure level is unreachable.
      if (pressure < pressure_range.min_pressure or
          pressure > pressure_range.max_pressure):
        feature_index = self._add_unreachable_pressure_levels(
            feature_vector, feature_index, 1)
        continue

      wind_vector = means[level, 0:2]
      magnitude = np.linalg.norm(wind_vector, ord=2)
      wind_vector /= (magnitude + TOLERANCE)

      # Compute the angle error (in radians, normalized in [0, 1]).
      # TODO(bellemare): The better alternative is to compute the angle of the
      # two vectors and subtract them. It still requires us to assign an angle
      # to the 0 vector but hides the logic somewhere better.
      # TODO(bellemare): Package this logic into a units function.
      if distance_to_station.meters < TOLERANCE:
        angle_error = 0.0
      elif magnitude < TOLERANCE:
        # TODO(bellemare): Check with Sal. How do we handle edge cases?
        angle_error = math.pi
      else:
        cos_angle_error = np.dot(wind_vector, station_direction)
        # Deal with numerical errors with np.clip.
        cos_angle_error = np.clip(cos_angle_error, -1, 1)
        angle_error = math.acos(cos_angle_error)
        assert 0 <= angle_error <= math.pi

      assert 0.0 <= deviations[level] <= 1.00001, 'Uncertainty not in [0, 1].'

      feature_vector[feature_index] = deviations[level]
      feature_vector[feature_index + 1] = (
          transforms.linear_rescale_with_extrapolation(angle_error, 0, math.pi))
      feature_vector[feature_index + 2] = transforms.squash_to_unit_interval(
          magnitude, 30)
      feature_index += 3

    # Pad the vector with unreachable data BELOW the valid range.
    feature_index = self._add_unreachable_pressure_levels(
        feature_vector, feature_index, num_levels_higher)

    # Make sure we allocated all elements of the feature vector.
    assert feature_index == len(feature_vector)

  # TODO(bellemare): Consider making this a @staticmethod.
  def _add_unreachable_pressure_levels(
      self, feature_vector: np.ndarray, index: int, levels: int) -> int:
    """Adds the given number of 'unreachable' pressure levels.

    These are set to (0, 1, 1), corresponding to being maximally confident
    that the wind is really, really bad. The first digit (0) indicates no
    uncertainty, the next digit (1) indicates the wind is blowing in the wrong
    direction, and the third digit (1) indicates the wind is blowing at
    maximum velocity.

    Args:
      feature_vector: the feature vector to fill up.
      index: the starting index in this feature vector.
      levels: the number of levels to add.

    Returns:
      the index in the feature vector after adding features.
    """
    for _ in range(levels):
      feature_vector[index:index + 3] = (0, 1, 1)
      index += 3

    return index
