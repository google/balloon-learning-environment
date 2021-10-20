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

"""Standard atmosphere computations.

These methods compute:
  - Temperature [K]
  - Pressure [Pa]
  - Density [kg/m^3]
  - Water vapor pressure [Pa]
  - Water vapor density [kg/m^3]
  - Radio refractivity index [.]
 given geopotential height [m].

 Temperature, pressure, and density based on U.S. Standard Atmosphere 1976
 described at:
     http://en.wikipedia.org/wiki/US_Standard_Atmosphere
     http://www.digitaldutch.com/atmoscalc/table.htm

 Water vapor pressure and density based on ITU-R P.835-5: "Reference Standard
 Atmospheres" available here:
     http://www.itu.int/rec/R-REC-P.835-5-201202-I

 Global radio refractivity index based on ITU-R P.453-10: "The radio
 refractive index: its formula and refractivity data" available here:
     http://www.itu.int/rec/R-REC-P.453-10-201202-I
"""
# TODO(joshgreavs): Maybe move this to balloon_learning_environment/env.

import dataclasses

from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import units
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class AtmosphericValues(object):
  """A dataclass containing the relevant values for standard atmosphere."""
  height: units.Distance
  temperature: float  # In Kelvin
  pressure: float  # In Pascals
  density: float  # In kg/m^3


class Atmosphere:
  """Atmospheric conditions for a variety of geographical locations.

  This is built on top of standard atmosphere, but includes lapse rates
  to simulate a broader range of atmospheric conditions.
  """
  _HEIGHT_TRANSITIONS = np.array(
      [-610.0, 17000.0, 21000.0, 32000.0, 47000.0, 51000.0, 71000.0, 85000.0])
  _LAPSE_RATES_LOW = np.array(
      [-0.007, 0.006, 0.001, 0.0028, 0.0, -0.0028, -0.002])
  _LAPSE_RATES_HIGH = np.array(
      [-0.0058, 0.005, 0.001, 0.0028, 0.0, -0.0028, -0.002])

  def __init__(self, key: jnp.ndarray):
    self.reset(key)

  def reset(self, key: jnp.ndarray) -> None:
    """Resets and samples a new atmosphere.

    Args:
      key: A PRNG key to use for sampling a new atmosphere.
    """
    alpha = jax.random.uniform(key).item()
    self._lapse_rates = ((1 - alpha) * self._LAPSE_RATES_LOW +
                         alpha * self._LAPSE_RATES_HIGH)

    self._initialize_temperature_transitions()
    self._initialize_pressure_transitions()

  def at_height(self, height: units.Distance) -> AtmosphericValues:
    """Computes atmosphere values at a specific height."""
    # We "unwrap" height into meters for legibility in the function.
    height = height.meters
    # Check that height is within expected range, specified by first and last
    # height transitions.
    assert height >= self._HEIGHT_TRANSITIONS[0]
    assert height < self._HEIGHT_TRANSITIONS[-1]

    # Compute standard atmosphere temperature and pressure given height.
    temperature = 0.0
    pressure = 0.0
    for i in range(len(self._lapse_rates)):
      # Check if height is within this range.
      if height < self._HEIGHT_TRANSITIONS[i + 1]:
        # Propagate temperature.
        temperature = self._temperature_transitions[i] + self._lapse_rates[i] * (
            height - self._HEIGHT_TRANSITIONS[i])
        # Propagate pressure.
        if self._lapse_rates[i] == 0.0:
          pressure = self._pressure_for_constant_temperature(
              height - self._HEIGHT_TRANSITIONS[i], temperature,
              self._pressure_transitions[i])
        else:
          pressure = self._pressure_for_linear_temperature(
              temperature / self._temperature_transitions[i],
              self._lapse_rates[i], self._pressure_transitions[i])
        break

    density = pressure / (constants.DRY_AIR_SPECIFIC_GAS_CONSTANT * temperature)
    return AtmosphericValues(
        units.Distance(meters=height), temperature, pressure, density)

  def at_pressure(self, pressure: float) -> AtmosphericValues:
    """Computes atmosphere values at a specific pressure."""
    # Check that pressure is within expected range, specified by last and first
    # pressure transitions.
    assert pressure > self._pressure_transitions[-1]
    assert pressure <= self._pressure_transitions[0]

    # Compute standard atmosphere temperature and height given pressure.
    temperature = 0.0
    height = 0.0
    for i in range(len(self._lapse_rates)):
      # Check if pressure is within this range.
      if pressure > self._pressure_transitions[i + 1]:
        # Compute height.
        if self._lapse_rates[i] == 0.0:
          height = ((-constants.DRY_AIR_SPECIFIC_GAS_CONSTANT *
                     self._temperature_transitions[i] / constants.GRAVITY) *
                    np.log(pressure / self._pressure_transitions[i]) +
                    self._HEIGHT_TRANSITIONS[i])
        else:
          height = (((pressure / self._pressure_transitions[i])**
                     (-constants.DRY_AIR_SPECIFIC_GAS_CONSTANT *
                      self._lapse_rates[i] / constants.GRAVITY) - 1) *
                    self._temperature_transitions[i] / self._lapse_rates[i] +
                    self._HEIGHT_TRANSITIONS[i])
        # Propagate temperature.
        temperature = self._temperature_transitions[i] + self._lapse_rates[i] * (
            height - self._HEIGHT_TRANSITIONS[i])
        break

    density = pressure / (constants.DRY_AIR_SPECIFIC_GAS_CONSTANT * temperature)
    return AtmosphericValues(
        units.Distance(meters=height), temperature, pressure, density)

  def _initialize_temperature_transitions(self) -> None:
    self._temperature_transitions = [300.0]  # Base temperature (in K).
    for i in range(len(self._lapse_rates)):
      self._temperature_transitions.append(
          self._temperature_transitions[-1] + self._lapse_rates[i] *
          (self._HEIGHT_TRANSITIONS[i + 1] - self._HEIGHT_TRANSITIONS[i]))
    self._temperature_transitions = np.array(self._temperature_transitions)

  def _initialize_pressure_transitions(self) -> None:
    """Initializes pressure transitions."""
    # We need temperature transitions initialized.
    self._pressure_transitions = [108870.8213]  # Base pressure (in Pa).
    for i in range(len(self._lapse_rates)):
      if self._lapse_rates[i] == 0.0:
        # Use constant temperature equation.
        self._pressure_transitions.append(
            self._pressure_for_constant_temperature(
                self._HEIGHT_TRANSITIONS[i + 1] - self._HEIGHT_TRANSITIONS[i],
                self._temperature_transitions[i + 1],
                self._pressure_transitions[-1]))
      else:
        # Use linear temperature equation.
        self._pressure_transitions.append(
            self._pressure_for_linear_temperature(
                self._temperature_transitions[i + 1] /
                self._temperature_transitions[i], self._lapse_rates[i],
                self._pressure_transitions[-1]))
    self._pressure_transitions = np.array(self._pressure_transitions)

  @staticmethod
  def _pressure_for_constant_temperature(delta_height: float,
                                         temperature: float,
                                         pressure_init: float) -> float:
    """Compute pressure for regions of constant temperature."""
    return pressure_init * np.exp(
        -(constants.GRAVITY * delta_height) /
        (constants.DRY_AIR_SPECIFIC_GAS_CONSTANT * temperature))

  @staticmethod
  def _pressure_for_linear_temperature(temperature_ratio: float,
                                       lapse_rate: float,
                                       pressure_init: float) -> float:
    """Compute pressure for regions of linearly changing temperature."""
    return pressure_init * (
        temperature_ratio
        **(-constants.GRAVITY /
           (constants.DRY_AIR_SPECIFIC_GAS_CONSTANT * lapse_rate)))
