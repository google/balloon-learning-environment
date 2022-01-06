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

"""Functions for calculating accessible pressure range for balloons."""

import dataclasses
import operator
from typing import Sequence

from balloon_learning_environment.env.balloon import altitude_safety
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import envelope_safety
from balloon_learning_environment.env.balloon import stable_init
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import constants
import numpy as np
from scipy import interpolate


@dataclasses.dataclass
class AccessiblePressureRange:
  min_pressure: float
  max_pressure: float


def _assert_monotonically_increasing(x: Sequence[float]):
  for i in range(1, len(x)):
    assert x[i] > x[i - 1]


def _compute_x_crossing(
    x1: float, y1: float, x2: float, y2: float, y_star: float) -> float:
  """Approximates the x value at which f(x) == y*.

  Args:
    x1: An x value.
    y1: f(x1).
    x2: An x value, where x2 > x1.
    y2: f(x2).
    y_star: The search value in [y1, y2].

  Returns:
    The x value that gives f(x) = y*, calculated with a linear approximation.
  """
  if y_star < min(y1, y2) or y_star > max(y1, y2):
    raise ValueError('y_star must be in [y1, y2].')
  if x1 >= x2:
    raise ValueError('x2 must be greater than x1.')
  if y1 == y2:
    raise ValueError('y1 may not be equal to y2.')

  alpha = abs((y_star - y1) / (y2 - y1))
  return alpha * (x2 - x1) + x1


def _compute_safe_pressure(
    pressure1: float, sp1: float, pressure2: float, sp2: float,
    min_sp: float, max_sp: float) -> float:
  """Computes the max/min pressure in range that has valid superpressure.

  This corresponds to a safety threshold crossing for superpressure
  (e.g. min or max superpressure).

  Args:
    pressure1: A first pressure value.
    sp1: The superpressure at pressure1.
    pressure2: A second pressure value. Assumes pressure2 > pressure.
    sp2: The superpressure at pressure2. Assumes sp2 != sp1.
    min_sp: The minimum safe superpressure.
    max_sp: The maximum safe superpressure.

  Returns:
    The pressure in [pressure1, pressure2] where the corresponding
      superpressure crosses the safety threshold (max or min superpressure).
  """
  if pressure1 >= pressure2:
    raise ValueError('pressure2 must be greater than pressure1.')
  if sp1 == sp2:
    raise ValueError('sp1 and sp2 may not be equal.')

  if (sp1 < min_sp and sp2 >= min_sp) or (sp1 >= min_sp and sp2 < min_sp):
    # Zero superpressure crossing.
    return _compute_x_crossing(pressure1, sp1, pressure2, sp2, min_sp)
  if ((sp1 > max_sp and sp2 <= max_sp) or
      (sp1 <= max_sp and sp2 > max_sp)):
    # Max superpressure crossing.
    return _compute_x_crossing(pressure1, sp1, pressure2, sp2, max_sp)
  # If neither of these happened then something failed.
  raise ValueError(
      'Unable to find valid superpressure crossing for input params.')


def _search_for_safe_pressure(b: balloon.BalloonState,
                              atmosphere: standard_atmosphere.Atmosphere,
                              pressure_column: Sequence[float],
                              significant_pressure: float,
                              min_superpressure: float,
                              max_superpressure: float,
                              direction: str) -> float:
  """Searches for a safe superpressure.

  Args:
    b: A balloon to search for a safe superpressure for.
    atmosphere: The current atmospheric conditions.
    pressure_column: A sequence of pressure levels to search at.
    significant_pressure: A significant pressure value. If direction == 'min',
      then this is the minimum (potentially unsafe) pressure that the balloon
      may float at. If direction == 'max', then this should be the maximum
      (potentially unsafe) pressure that the balloon may float at.
    min_superpressure: The minimum safe superpressure.
    max_superpressure: The maximum safe superpressure.
    direction: 'min' to search for the lowest pressure level with safe
      superpressure, or 'max' to search for the highest pressure level
      with safe superpressure.

  Returns:
    The maximum (direction == 'max') or minimum (direction == 'min')
      pressure [Pa] that the balloon can float at while maintaining a safe
      superpressure.
  """
  stable_params = stable_init.calculate_stable_params_for_pressure(
      significant_pressure, b.envelope_volume_base,
      b.envelope_volume_dv_pressure, b.envelope_mass, b.payload_mass,
      b.mols_lift_gas, b.latlng, b.date_time, b.upwelling_infrared, atmosphere)
  sp = stable_params.superpressure
  if min_superpressure <= sp <= max_superpressure:
    # If significant_pressure is safe, return it (it is max or min).
    return significant_pressure

  # Otherwise, begin search.
  # We check every point in the pressure column to see if it's within valid
  # superpressure limits or not. We only consider plausible points (i.e.
  # within [P_top, P_bottom]).
  last_pressure_sp = (significant_pressure, sp)
  if direction == 'min':
    pressure_column = reversed(pressure_column)
    comparator = operator.gt
  elif direction == 'max':
    comparator = operator.lt
  else:
    raise ValueError(f'Unknown value for direction {direction}')

  for pressure in pressure_column:
    if comparator(pressure, significant_pressure):
      continue  # Point is outside valid altitude range. Skip to next.

    stable_params = stable_init.calculate_stable_params_for_pressure(
        pressure, b.envelope_volume_base, b.envelope_volume_dv_pressure,
        b.envelope_mass, b.payload_mass, b.mols_lift_gas, b.latlng, b.date_time,
        b.upwelling_infrared, atmosphere)
    sp = stable_params.superpressure

    if sp > max_superpressure or sp < min_superpressure:
      # This pressure is not safe. Update variables and keep searching.
      last_pressure_sp = (pressure, sp)
      continue

    # Found our first safe pressure. Initialize the top range by
    # interpolating between this safe pressure and the last unsafe pressure.
    if direction == 'min':
      return _compute_safe_pressure(
          pressure, sp, last_pressure_sp[0], last_pressure_sp[1],
          min_superpressure, max_superpressure)
    else:
      return _compute_safe_pressure(
          last_pressure_sp[0], last_pressure_sp[1],
          pressure, sp, min_superpressure, max_superpressure)

  # If no safe pressure was found, raise an error.
  raise ValueError('Unable to find safe pressure for balloon.')


def _find_min_pressure_with_safe_superpressure(
    b: balloon.BalloonState, atmosphere: standard_atmosphere.Atmosphere,
    pressure_levels: Sequence[float], max_altitude_pressure: float,
    min_superpressure: float, max_superpressure: float):
  return _search_for_safe_pressure(b, atmosphere, pressure_levels,
                                   max_altitude_pressure, min_superpressure,
                                   max_superpressure, 'max')


def _find_max_pressure_with_safe_superpressure(
    b: balloon.BalloonState, atmosphere: standard_atmosphere.Atmosphere,
    pressure_levels: Sequence[float], min_altitude_pressure: float,
    min_superpressure: float, max_superpressure: float):
  return _search_for_safe_pressure(b, atmosphere, pressure_levels,
                                   min_altitude_pressure, min_superpressure,
                                   max_superpressure, 'min')


def get_pressure_range(
    b: balloon.BalloonState,
    atmosphere: standard_atmosphere.Atmosphere) -> AccessiblePressureRange:
  """Gets the range of accessible pressures.

  Args:
    b: a balloon to find the accessible pressure range for.
    atmosphere: The atmospheric conditions the balloon is flying in.

  Returns:
    The accessible pressure range.
  """
  # TODO(joshgreaves): Check the parameter conversions with Sal.
  # mols_gas_in_envelope => mols_lift_gas
  # gas_in_envelope_molar_mass => HE_MOLAR_MASS
  # mass_system => payload_mass + envelope_mass
  # volume_total => envelope_volume_base (not envelope_volume)?
  # current_internal_gas_temperature => internal_temperature
  # current_ambient_pressure => pressure
  # max_superpressure => envelope_max_pressure

  superpressure_buffer = envelope_safety.BUFFER
  min_superpressure_with_buffer = superpressure_buffer
  max_superpressure_with_buffer = (
      b.envelope_max_superpressure - superpressure_buffer)
  assert max_superpressure_with_buffer > 0.0

  search_range_min = 1000.0
  search_range_max = atmosphere.at_height(altitude_safety.MIN_ALTITUDE).pressure
  pressure_levels = np.linspace(search_range_min, search_range_max, 20)
  column = [atmosphere.at_pressure(x) for x in pressure_levels]

  # Compute mass of the system when ballonet empty.
  total_empty_mass = (b.payload_mass + b.envelope_mass +
                      b.mols_lift_gas * constants.HE_MOLAR_MASS)

  # Compute pressure / temperature (P/T) for balloon floating at maximum
  # altitude (mols_air = 0). We use the equation:
  #    P_amb / T_amb = mass * R / (M_air * V)
  # Note: use envelope_volume_base for volume when mols_air = 0.
  max_altitude_p_over_t = (
      total_empty_mass * constants.UNIVERSAL_GAS_CONSTANT /
      (constants.DRY_AIR_MOLAR_MASS * b.envelope_volume_base))

  # Find min pressure (max altitude) by interpolating from the table of
  # P/T vs P. If outside the range, extend the projection.
  # TODO(joshgreaves): How accurate is this? It only takes into account ambient
  # temperature, but gas temperature may be quite different during daytime.
  p_over_t_column = [x.pressure / x.temperature for x in column]
  _assert_monotonically_increasing(p_over_t_column)
  min_pressure = interpolate.interp1d(
      p_over_t_column, pressure_levels, kind='linear',
      fill_value='extrapolate')(max_altitude_p_over_t).item()

  # Initialize pressure at bottom of range to hard-coded max.
  # TODO(joshgreaves): The max pressure is the pressure that corresponds to
  # the altitude floor in altitude_safety.
  max_pressure = search_range_max

  # Compute max and min pressures that have safe superpressure.
  min_safe_pressure = _find_min_pressure_with_safe_superpressure(
      b, atmosphere, pressure_levels, min_pressure,
      min_superpressure_with_buffer, max_superpressure_with_buffer)
  max_safe_pressure = _find_max_pressure_with_safe_superpressure(
      b, atmosphere, pressure_levels, max_pressure,
      min_superpressure_with_buffer, max_superpressure_with_buffer)

  # Ensure we are return floats, rather than np.floats.
  min_safe_pressure = float(min_safe_pressure)
  max_safe_pressure = float(max_safe_pressure)
  return AccessiblePressureRange(
      min_pressure=min_safe_pressure,
      max_pressure=max_safe_pressure)
