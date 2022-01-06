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

"""Calculates stable parameters for balloon initialization."""

import dataclasses
import datetime as dt

from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon import thermal
from balloon_learning_environment.utils import constants
import numpy as np

import s2sphere as s2


@dataclasses.dataclass
class StableParams:
  ambient_temperature: float
  internal_temperature: float
  mols_air: float
  envelope_volume: float
  superpressure: float


def calculate_stable_params_for_pressure(
    pressure: float, envelope_volume_base: float,
    envelope_volume_dv_pressure: float, envelope_mass: float,
    payload_mass: float, mols_lift_gas: float, latlng: s2.LatLng,
    date_time: dt.datetime, upwelling_infrared: float,
    atmosphere: standard_atmosphere.Atmosphere) -> StableParams:
  """Calculates stable parameter values for the ambient pressure.

  This calculates the internal and external temperature for a balloon
  at the specified pressure, as well as the mols air in the ballonet,
  envelope volume, and superpressure required to float at the specified
  ambient temperature.

  Args:
    pressure: Ambient pressure of the balloon [Pa].
    envelope_volume_base: The y-intercept for the balloon envelope volume
      model [m^3].
    envelope_volume_dv_pressure: The slope for the balloon envelope volume
      model.
    envelope_mass: Mass of the balloon envelope [kg].
    payload_mass: The mass of the payload. The term payload here refers to
      all parts of the flight system other than the balloon envelope [kg].
    mols_lift_gas: Mols of helium within the balloon envelope [mols].
    latlng: The current latitude and longitude of the balloon.
    date_time: The current date and time of the balloon.
    upwelling_infrared: The upwelling infrared value.
    atmosphere: The current atmosphere state.

  Returns:
    A tuple of (ambient temperature [K], mols air in ballonet [mols]).
  """
  ambient_temperature = atmosphere.at_pressure(pressure).temperature
  # ---- Cold start mols air in envelope ----
  # Compute the mols gas in balloon that gives the desired pressure.
  # This comes from rho * V = m, where:
  #
  #.         ambient_pressure * air_molar_mass
  #   rho =  ----------------------------------
  #          universal_gas_const * ambient_temp
  #
  #   m = (mass_envelope + mass_payload +
  #        helium_molar_mass * mols_helium +
  #        air_molar_mass * mols_air)
  #
  # Then, you just solve for mols_air to get the following equation.
  mols_air = (
      (pressure * constants.DRY_AIR_MOLAR_MASS * envelope_volume_base /
       (constants.UNIVERSAL_GAS_CONSTANT * ambient_temperature) -
       envelope_mass - payload_mass - constants.HE_MOLAR_MASS * mols_lift_gas)
      / constants.DRY_AIR_MOLAR_MASS)
  # TODO(joshgreaves): Warning or Exception for initializing out of range?
  mols_air = np.clip(mols_air, 0.0, None)

  # ---- Cold start internal temperature ----
  internal_temperature = 206.0  # [K] pick an average value to start search.
  solar_elevation, _, solar_flux = solar.solar_calculator(latlng, date_time)

  # Apply a few iterations of Newton-Raphson to find where the rate of
  # change of temperature is close to 0.
  delta_temp = 0.01
  for _ in range(10):
    # Note: we use envelope_volume_base rather than envelope_volume, since
    # empirically it doesn't make much of a difference, and the envelope
    # volume isn't calculated until the temperature is calculated.
    d_internal_temp1 = thermal.d_balloon_temperature_dt(
        envelope_volume_base, envelope_mass,
        internal_temperature - delta_temp / 2, ambient_temperature, pressure,
        solar_elevation, solar_flux, upwelling_infrared)
    d_internal_temp2 = thermal.d_balloon_temperature_dt(
        envelope_volume_base, envelope_mass,
        internal_temperature + delta_temp / 2, ambient_temperature, pressure,
        solar_elevation, solar_flux, upwelling_infrared)

    # d2_internal_temp is the second derivitive of temperature w.r.t time.
    d2_internal_temp = (d_internal_temp2 - d_internal_temp1) / delta_temp
    mean_d_internal_temp = (d_internal_temp1 + d_internal_temp2) / 2.0
    if abs(d2_internal_temp) > 0.0:
      internal_temperature -= (mean_d_internal_temp / d2_internal_temp)

    if abs(mean_d_internal_temp) < 1e-5:
      break

  # ---- Cold start superpressure ----
  envelope_volume, superpressure = (
      balloon.Balloon.calculate_superpressure_and_volume(
          mols_lift_gas, mols_air, internal_temperature, pressure,
          envelope_volume_base, envelope_volume_dv_pressure))

  return StableParams(ambient_temperature, internal_temperature, mols_air,
                      envelope_volume, superpressure)


def cold_start_to_stable_params(
    balloon_state: balloon.BalloonState,
    atmosphere: standard_atmosphere.Atmosphere) -> None:
  """Sets parameters to stable values for the ambient pressure.

  The pressure altitude of the balloon depends on a number of variables,
  such as the number of mols of air in the ballonet, the temperature
  of air and gas inside the envelope, and the superpressure. To have
  a balloon float at a specific pressure level, these parameters should
  be updated to match the specified ambient pressure.

  Args:
    balloon_state: The balloon state to update with stable params.
    atmosphere: The current atmosphere the balloon is flying in.
  """
  stable_params = calculate_stable_params_for_pressure(
      balloon_state.pressure, balloon_state.envelope_volume_base,
      balloon_state.envelope_volume_dv_pressure, balloon_state.envelope_mass,
      balloon_state.payload_mass, balloon_state.mols_lift_gas,
      balloon_state.latlng, balloon_state.date_time,
      balloon_state.upwelling_infrared, atmosphere)
  balloon_state.ambient_temperature = stable_params.ambient_temperature
  balloon_state.internal_temperature = stable_params.internal_temperature
  balloon_state.mols_air = stable_params.mols_air
  balloon_state.envelope_volume = stable_params.envelope_volume
  balloon_state.superpressure = stable_params.superpressure
