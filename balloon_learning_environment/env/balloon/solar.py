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

"""Utilities for modeling sun-related variables.

This includes solar power and also sunrise/sunset related time calculations.
"""

import collections
import datetime as dt
import functools
import math
import operator
from typing import Callable, Tuple, Union

from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import units
import numpy as np

import s2sphere as s2

# We define minimum solar elevation based on Earth's radius and mean balloon
# altitude. Balloons typically fly from 15km-20km, so we assume 17.5km.
# If solar elevation is below min solar horizon we can't see the sun anymore.
# NOTE: Angle is below horizon so we need a negative sign.
MIN_SOLAR_EL_DEG = -4.242
_SEARCH_TIME_DELTA = dt.timedelta(minutes=3)


# TODO(joshgreaves): Use s2.S1Angle throughout.
def solar_calculator(latlng: s2.LatLng,
                     time: dt.datetime) -> Tuple[float, float, float]:
  """Computes solar elevation, azimuth, and flux given latitude/longitude/time.

     Based on NOAA Solar Calculator described at:
       http://www.esrl.noaa.gov/gmd/grad/solcalc/index.html

  Args:
    latlng: The latitude and longitude.
    time: Datetime object.

  Returns:
    el_deg: Solar elevation in degrees.
    az_deg: Solar azimuth in degrees.
    flux: Solar flux in W/m^2.
  """
  # Check if latitude is within expected range.
  if not latlng.is_valid:
    raise ValueError(f'solar_calculator: latlng is invalid: {latlng}.')
  if time.tzinfo is None:
    raise ValueError('time parameter needs timezone. Try UTC.')

  # Compute fraction_of_day from time.
  fraction_of_day = (
      int(time.timestamp()) %
      constants.NUM_SECONDS_PER_DAY) / constants.NUM_SECONDS_PER_DAY

  # Compute Julian day number from Gregorian calendar.
  julian_day_number = (367.0 * time.year - np.floor(7.0 * (time.year + np.floor(
      (time.month + 9.0) / 12.0)) / 4.0) - np.floor(3.0 * (np.floor(
          (time.year + (time.month - 9.0) / 7.0) / 100.0) + 1.0) / 4.0) +
                       np.floor(275.0 * time.month / 9.0) + time.day +
                       1721028.5)

  # Compute Julian time (in days and in centuries).
  julian_time = julian_day_number + fraction_of_day
  julian_century = (julian_time - 2451545.0) / 36525.0

  # Compute solar parameters.
  geometric_mean_long_sun = math.radians(
      280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032))
  sin2l0 = np.sin(2.0 * geometric_mean_long_sun)
  cos2l0 = np.cos(2.0 * geometric_mean_long_sun)
  sin4l0 = np.sin(4.0 * geometric_mean_long_sun)

  geometric_mean_anomaly_sun = math.radians(
      357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century))
  sinm0 = np.sin(geometric_mean_anomaly_sun)
  sin2m0 = np.sin(2.0 * geometric_mean_anomaly_sun)
  sin3m0 = np.sin(3.0 * geometric_mean_anomaly_sun)

  mean_obliquity_of_ecliptic = math.radians(23.0 + (26.0 + (
      (21.448 - julian_century *
       (46.815 + julian_century *
        (0.00059 - julian_century * 0.001813)))) / 60.0) / 60.0)

  obliquity_correction = mean_obliquity_of_ecliptic + math.radians(
      0.00256 * np.cos(math.radians(125.04 - 1934.136 * julian_century)))

  var_y = np.tan(obliquity_correction / 2.0)**2

  eccentricity_earth = 0.016708634 - julian_century * (
      0.000042037 + 0.0000001267 * julian_century)

  equation_of_time = (4.0 *
                      (var_y * sin2l0 - 2.0 * eccentricity_earth * sinm0 +
                       4.0 * eccentricity_earth * var_y * sinm0 * cos2l0 -
                       0.5 * var_y * var_y * sin4l0 -
                       1.25 * eccentricity_earth * eccentricity_earth * sin2m0))

  hour_angle = math.radians(
      math.fmod(
          1440.0 * fraction_of_day + math.degrees(equation_of_time) +
          4.0 * latlng.lng().degrees, 1440.0)) / 4.0
  if hour_angle < 0:
    hour_angle += math.pi
  else:
    hour_angle -= math.pi

  eq_of_center_sun = math.radians(sinm0 *
                                  (1.914602 - julian_century *
                                   (0.004817 + 0.000014 * julian_century)) +
                                  sin2m0 *
                                  (0.019993 - 0.000101 * julian_century) +
                                  sin3m0 * 0.000289)
  true_long_sun = geometric_mean_long_sun + eq_of_center_sun
  apparent_long_sun = true_long_sun - math.radians(
      0.00569 -
      0.00478 * np.sin(math.radians(125.04 - 1934.136 * julian_century)))
  declination_sun = np.arcsin(
      np.sin(obliquity_correction) * np.sin(apparent_long_sun))

  zenith_angle = np.arccos(
      np.sin(latlng.lat().radians) * np.sin(declination_sun) +
      np.cos(latlng.lat().radians) * np.cos(declination_sun) *
      np.cos(hour_angle))

  # Compute solar elevation. Correct for atmospheric refraction.
  el_uncorrected_deg = 90.0 - math.degrees(zenith_angle)

  if el_uncorrected_deg > 85.0:
    atmospheric_refraction = 0
  elif el_uncorrected_deg > 5.0:
    tan_seu = np.tan(math.radians(el_uncorrected_deg))
    atmospheric_refraction = (58.1 / tan_seu - 0.07 / (tan_seu**3) + 0.000086 /
                              (tan_seu**5))
  elif el_uncorrected_deg > -0.575:
    atmospheric_refraction = (1735.0 + el_uncorrected_deg *
                              (-518.2 + el_uncorrected_deg *
                               (103.4 + el_uncorrected_deg *
                                (-12.79 + el_uncorrected_deg * 0.711))))
  else:
    atmospheric_refraction = -20.772 / np.tan(math.radians(el_uncorrected_deg))

  el_deg = el_uncorrected_deg + atmospheric_refraction / 3600.0

  # Compute solar azimuth. Make sure cos_azimuth is in the range [-1, 1].
  cos_az = ((np.sin(latlng.lat().radians) * np.cos(zenith_angle) -
             np.sin(declination_sun)) /
            (np.cos(latlng.lat().radians) * np.sin(zenith_angle)))
  az_unwrapped = np.arccos(np.clip(cos_az, -1.0, 1.0))
  if hour_angle > 0:
    az_deg = math.degrees(az_unwrapped) + 180.0
  else:
    az_deg = 180.0 - math.degrees(az_unwrapped)

  # Compute solar flux in W/m^2.
  flux = 1366.0 * (1 + 0.5 * (
      ((1 + eccentricity_earth) /
       (1 - eccentricity_earth))**2 - 1) * np.cos(geometric_mean_anomaly_sun))

  return el_deg, az_deg, flux


def solar_atmospheric_attenuation(el_deg: float,
                                  pressure_altitude_pa: float) -> float:
  """Computes atmospheric attenuation of incoming solar radiation.

  Args:
    el_deg: Solar elevation in degrees.
    pressure_altitude_pa: Balloon's pressure altitude in Pascals.

  Returns:
    attenuation_factor: Solar atmospheric attenuation factor in range [0, 1].
  """

  # Check if solar elevation is within range [-90, 90] deg.
  if el_deg > 90.0 or el_deg < -90.0:
    raise ValueError('solar_atmospheric_attenuation: '
                     'Solar elevation out of expected range [-90, 90] deg.')

  # Check if pressure altitude [Pa] is within range [0, 101325] Pa.
  if pressure_altitude_pa > 101325.0 or pressure_altitude_pa < 0.0:
    raise ValueError('solar_atmospheric_attenuation: '
                     'Pressure altitude out of expected range [0, 101325] Pa.')

  # If solar elevation is below min solar horizon return 0.
  if el_deg < MIN_SOLAR_EL_DEG:
    return 0.0

  # Compute airmass.
  tmp_sin_elev = 614.0 * np.sin(math.radians(el_deg))
  airmass = (0.34764 * (pressure_altitude_pa / 101325.0) *
             (math.sqrt(1229.0 + tmp_sin_elev * tmp_sin_elev) - tmp_sin_elev))

  # Compute atmospheric attenuation factor.
  return 0.5 * (np.exp(-0.65 * airmass) + np.exp(-0.95 * airmass))


def balloon_shadow(el_deg: float, panel_height_below_balloon_m: float) -> float:
  """Computes shadowing factor on solar panels due to balloon film.

  Args:
    el_deg: Solar elevation in degrees.
    panel_height_below_balloon_m: Panel location below balloon in meters.

  Returns:
    shadow_factor: Balloon shadowing factor in range [0, 1].
  """
  balloon_radius = 8.69275
  balloon_height = 10.41603

  shadow_el_deg = math.degrees(
      np.arctan2(
          math.sqrt(panel_height_below_balloon_m *
                    (balloon_height + panel_height_below_balloon_m)),
          balloon_radius))

  if el_deg >= shadow_el_deg:
    # Shadowing applies. Use a balloon shadow factor of 0.4392.
    return 0.4392
  else:
    # No shadow.
    return 1.0


def is_solar_afternoon(latlng: s2.LatLng, time: dt.datetime) -> bool:
  """Returns whether it is the solar afternoon.

  That is, returns whether midnight will happen before noon (chronologically).

  Args:
    latlng: Latitude/longitude at which to calculate.
    time: Datetime at which to calculate.

  Returns:
    True if midnight is coming before noon.
  """
  now_elevation, _, _ = solar_calculator(latlng, time)
  then_elevation, _, _ = solar_calculator(
      latlng, time + dt.timedelta(seconds=1))

  return then_elevation < now_elevation


def _find_solar_elevation(
    latlng: s2.LatLng,
    min_time: dt.datetime,
    max_time: dt.datetime,
    target: Union[str, float],
    time_delta=_SEARCH_TIME_DELTA) -> Tuple[dt.datetime, float]:
  """A user-friendly wrapper around _find_solar_elevation_binary_search.

  See caveats in the comments to that function.

  Args:
    latlng: Latitude/longitude at which to calculate.
    min_time: Earliest time in the time interval to consider.
    max_time: Latest time in the time interval to consider.
    target: One of 'minimum', 'maximum', or a specific elevation to be located.
    time_delta: Resolution of search process. If None, use 3 minutes.

  Returns:
    time: time at which the next midnight (or noon) will occur.
    elevation: solar elevation at that time.
  """
  if target == 'minimum':
    return _find_solar_elevation_binary_search(
        latlng, min_time, max_time, operator.pos, time_delta)
  elif target == 'maximum':
    return _find_solar_elevation_binary_search(
        latlng, min_time, max_time, operator.neg, time_delta)
  else:
    try:
      # Turn the numerical value into an absolute loss function.
      target_numeric = float(target)
      return _find_solar_elevation_binary_search(
          latlng, min_time, max_time,
          lambda x: abs(x - target_numeric), time_delta)
    except ValueError:
      raise ValueError('Unknown target type: {}'.format(target))


def _find_solar_elevation_binary_search(
    latlng: s2.LatLng,
    min_time: dt.datetime,
    max_time: dt.datetime,
    transfer_function: Callable[[float], float],
    time_delta=_SEARCH_TIME_DELTA) -> Tuple[dt.datetime, float]:
  """Finds the next solar midnight or noon in the given time interval.

  This method assumes that on the given interval, the transfer_function results
  in a convex objective that can be minimized.

  Args:
    latlng: Latitude/longitude at which to calculate.
    min_time: Earliest time in the time interval to consider.
    max_time: Latest time in the time interval to consider.
    transfer_function: Transfer function to be applied to the elevation.
    time_delta: Resolution of search process. If None, use 3 minutes.

  Returns:
    time: time at which the next midnight (or noon) will occur.
    elevation: solar elevation at that time.
  """
  if max_time < min_time:
    raise ValueError('Time interval must have positive extent.')

  max_steps = int((max_time - min_time) / time_delta)
  assert max_steps > 0

  # This calculates the solar elevation at a fixed timestep in the future, as
  # an objective to be minimized.
  # idx is the number of timesteps in the future.
  # Assumes that all timesteps have the same length.
  def _objective_function(idx: int) -> float:
    time = min_time + time_delta * idx
    el_degree, _, _ = solar_calculator(latlng, time)

    # If looking for noon, we negate the elevation curve to minimize it.
    return transfer_function(el_degree)

  # TODO(bellemare): Move this somewhere?
  class _LazySequence(collections.abc.Sequence):
    """A Sequence that calculates its values on the fly."""

    def __init__(self, length: int, fn: Callable[[int], float]):
      self._len = length
      self._fn = fn

    @functools.lru_cache(maxsize=200)
    def __getitem__(self, idx: int):
      return self._fn(idx)

    def __len__(self) -> int:
      return self._len

  # Perform binary search on the interval. The transfer_function transforms
  # solar elevation into a convex objective.
  objective = _LazySequence(max_steps, _objective_function)

  low = 0
  high = max_steps

  # Binary search the function for its minimum.
  while high > low + 1:
    midpoint = low + (high - low) / 2
    if objective[low] < objective[high]:
      # This trick works when the function is symmetric around its minimum.
      high = math.ceil(midpoint)  # Ceil/floor is a bit more conservative.
    else:
      low = math.floor(midpoint)

  # If all went well, the minimum is either high or low.
  if objective[low] < objective[high]:
    min_index = low
  else:
    min_index = high

  time = min_time + time_delta * min_index

  elevation, _, _ = solar_calculator(latlng, time)
  return time, elevation


def get_next_solar_midnight(
    latlng: s2.LatLng, time: dt.datetime,
    time_delta=_SEARCH_TIME_DELTA) -> Tuple[dt.datetime, float]:
  """Determines the next time at which solar midnight will occur.

  We call solar midnight the time at which the sun is at its lowest elevation.

  Args:
    latlng: Latitude/longitude at which to calculate.
    time: Datetime at which to calculate.
    time_delta: Resolution at which to determine time.

  Returns:
    time: time at which the next midnight will occur.
    elevation: solar elevation at that midnight.
  """
  if is_solar_afternoon(latlng, time):
    # Midnight is in the next 12 hours.
    return _find_solar_elevation(
        latlng, time, time + dt.timedelta(hours=12), 'minimum', time_delta)
  else:
    # Midnight is 12 to 24 hours away.
    return _find_solar_elevation(
        latlng, time + dt.timedelta(hours=12),
        time + dt.timedelta(hours=24), 'minimum', time_delta)


def get_next_solar_noon(
    latlng: s2.LatLng, time: dt.datetime,
    time_delta=_SEARCH_TIME_DELTA) -> Tuple[dt.datetime, float]:
  """Determines the next time at which solar noon will occur.

  This is the same as get_next_solar_midnight, but with some bits flipped.
  We call solar noon the time at which the sun is at its highest elevation.

  Args:
    latlng: Latitude/longitude at which to calculate.
    time: Datetime at which to calculate.
    time_delta: Resolution at which to determine time.

  Returns:
    time: time at which the next midnight will occur.
    elevation: solar elevation at that midnight.
  """
  if is_solar_afternoon(latlng, time):
    # Noon is in the next 12 to 24 hours.
    return _find_solar_elevation(
        latlng, time + dt.timedelta(hours=12),
        time + dt.timedelta(hours=24), 'maximum', time_delta)
  else:
    return _find_solar_elevation(
        latlng, time, time + dt.timedelta(hours=12), 'maximum', time_delta)


def get_next_sunrise_sunset(
    latlng: s2.LatLng,
    time: dt.datetime,
    time_delta=_SEARCH_TIME_DELTA) -> Tuple[dt.datetime, dt.datetime]:
  """Determines the next sunrise and sunset times.

  Args:
    latlng: Latitude/longitude at which to calculate.
    time: Datetime at which to calculate.
    time_delta: Resolution at which to determine time.

  Returns:
    sunrise: Time of next sunrise.
    sunset: Time of next sunset.
  """
  # This avoids dealing with polar day/night.
  # TODO(joshgreaves): Decide if we want to deal with the polar cases.
  assert abs(latlng.lat().degrees) < 60.0, 'High latitudes not supported.'

  next_noon, _ = get_next_solar_noon(latlng, time, time_delta)
  next_midnight, _ = get_next_solar_midnight(latlng, time, time_delta)

  # There are four cases here (four quadrant of the solar day). The next
  # lines pick out two of these quadrants. They might return a time before
  # the current time, in which case we know the relevant sunrise or sunset
  # is one day ahead.
  if is_solar_afternoon(latlng, time):
    # 'next_noon' is tomorrow.
    sunrise = _find_solar_elevation(
        latlng, next_midnight, next_noon,
        MIN_SOLAR_EL_DEG, time_delta)[0]
    sunset = _find_solar_elevation(
        latlng,
        next_noon - dt.timedelta(days=1), next_midnight,
        MIN_SOLAR_EL_DEG, time_delta)[0]
  else:
    # 'next_noon' is today.
    sunrise = _find_solar_elevation(
        latlng,
        next_midnight - dt.timedelta(days=1), next_noon,
        MIN_SOLAR_EL_DEG, time_delta)[0]
    sunset = _find_solar_elevation(
        latlng, next_noon, next_midnight,
        MIN_SOLAR_EL_DEG, time_delta)[0]

  # Handle the post-sunrise and post-sunset quadrants.
  if sunrise < time:
    sunrise += dt.timedelta(days=1)
  if sunset < time:
    sunset += dt.timedelta(days=1)

  return sunrise, sunset  # Swiftly flow the days.


def calculate_steps_to_sunrise(latlng: s2.LatLng,
                               time: dt.datetime,
                               time_delta=_SEARCH_TIME_DELTA) -> int:
  """Calculates the number of steps to next sunrise.

  When the number of steps is fractional, this is rounded up.

  Args:
    latlng: Latitude/longitude at which to calculate.
    time: Datetime at which to calculate.
    time_delta: The amount of time between each action.

  Returns:
    The number of time steps of length time_delta until sunrise. If the
    sun is up, returns 0.
  """
  # It's currently day, 0 steps.
  now_elevation, _, _ = solar_calculator(latlng, time)
  if now_elevation >= MIN_SOLAR_EL_DEG:
    return 0

  sunrise, _ = get_next_sunrise_sunset(latlng, time, time_delta)

  elapsed_time = sunrise - time
  elapsed_time_in_steps = math.ceil(elapsed_time / time_delta)

  return int(elapsed_time_in_steps)


def solar_power(el_deg: float, pressure_altitude_pa: float) -> units.Power:
  """Computes solar power produced by panels on the balloon.

  Args:
    el_deg: Solar elevation in degrees.
    pressure_altitude_pa: Balloon's pressure altitude in Pascals.

  Returns:
    solar_power: Solar power from panels on the balloon [W].
  """

  # Get atmospheric attenuation factor.
  attenuation = solar_atmospheric_attenuation(el_deg, pressure_altitude_pa)

  # Loon balloons have 4 main solar panels mounted at 35deg and hanging at 3.3m
  # below the balloon. There are an additional 2 panels mounted at 65deg
  # hanging at 2.7m below the balloon. All panels have a max power of 210 W.
  power = 210.0 * attenuation * (
      4 * np.cos(math.radians(el_deg - 35)) * balloon_shadow(el_deg, 3.3) +
      2 * np.cos(math.radians(el_deg - 65)) * balloon_shadow(el_deg, 2.7))

  return units.Power(watts=power)
