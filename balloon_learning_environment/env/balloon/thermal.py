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

"""Utilities for modeling balloon temperature."""

from balloon_learning_environment.env.balloon import solar
import numpy as np

# View factors are used in thermal modeling and refer to the percentage of the
# balloon involved in the thermal heat transfer (normalized effective area).
# Solar view factor is ~25% since the sun is really far away and can be treated
# as a directional point source acting on a sphere. Earth view factor is ~50%
# since the bottom half of the balloon can see the Earth (we use a slightly
# lower value that assumes the earth is a really large sphere rather than a
# flat plane).
_SOLAR_VIEW_FACTOR = 0.25
_EARTH_VIEW_FACTOR = 0.4605

# PE01 film params.
# - Reflectivity is how much light bounces off the film. It is assumed
#   to be the same for all wavelengths (solar and IR).
# - Solar absorptivity is the film absorption in the solar spectrum.
# - Absorptivity in the IR spectrum is a function of IR black-body
#   temperature. This is modeled as a linear function around a trim
#   condition:
#     A_ir = A_ir_base + A_ir_d_dT * (T_object - T_ref)
# - Specific heat of film.
_PE01_REFLECTIVITY = 0.0291
_PE01_ABSORPTIVITY_SOLAR = 0.01435
_PE01_ABSORPTIVITY_IR_BASE = 0.04587
_PE01_ABSORPTIVITY_IR_D_TEMPERATURE = 0.000232  # [1/K]
_PE01_ABSORPTIVITY_IR_REF_TEMPERATURE = 210  # [K]
_PE01_FILM_SPECIFIC_HEAT = 1500  # [J/(kg.K)]

_STEFAN_BOLTZMAN = 0.000000056704  # [W/(m^2.K^4)]
_UNIVERSAL_GAS_CONSTANT = 8.3144621  # [J/(mol.K)]
_DRY_AIR_MOLAR_MASS = 0.028964922481160  # Dry Air. [kg/mol]


def black_body_temperature_to_flux(temperature_k: float) -> float:
  """Compute corresponding flux given black body temperature.

  Args:
    temperature_k: Black body's temperature [K].

  Returns:
    flux: Black body's flux [W/m^2].
  """
  return _STEFAN_BOLTZMAN * temperature_k**4


def black_body_flux_to_temperature(flux: float) -> float:
  """Compute corresponding temperature given black body flux.

  Args:
    flux: Black body's flux [W/m^2].

  Returns:
    temperature_k: Black body's temperature [K].
  """
  return (flux / _STEFAN_BOLTZMAN)**0.25


def absorptivity_ir(object_temperature_k: float) -> float:
  """Compute balloon IR absorptivity/emissivity given black body temperature.

  This function computes IR absorptivity/emissivity given the radiative object's
  black body temperature. We assume PE01 balloon film and use a linear model.

  Args:
    object_temperature_k: Object's black body temperature [K].

  Returns:
    absorptivity: Absorptivity factor in IR spectrum.
  """
  return (_PE01_ABSORPTIVITY_IR_BASE + _PE01_ABSORPTIVITY_IR_D_TEMPERATURE *
          (object_temperature_k - _PE01_ABSORPTIVITY_IR_REF_TEMPERATURE))


def total_absorptivity(absorptivity: float, reflectivity: float) -> float:
  """Compute total internal balloon absorptivity/emissivity factor.

  This function computes total absorptivity or total emissivity. For the
  absorptivity process, the dynamics are as follows:

  --> From the radiation hitting the surface, R is reflected outward, A is
      absorbed, and the rest, T, is "transmitted" through the surface,
      where T = 1 - R - A.
  --> From the amount transmitted through the surface, T, a portion TA is
      absorbed, a portion TR is reflected back into the sphere, and the
      rest (TT) leaves the sphere.
  --> From the amount reflected into the sphere, TR, a portion TRA is
      absorbed, a portion TRR is reflected back into the sphere, and the
      rest (TRT) is lost.
  --> Continuing this process to infinity and adding up all the aborbed
      amounts gives the following:

         A_total = A + TA + TRA + TR^2 A + TR^3 A + ...
                 = A + TA (1 + R + R^2 + R^3 + ...)
                 = A (1 + T / (1 - R))

  Similarly, we can analyze the emissivity process:

  --> From the radiation emitted by the surface, A is emitted outwards where
      E = A = emissivity, and A is emitted inwards (double radiation).
  --> From the inwards amount, AA is re-absorbed, AR is internally reflected
      and AT is emitted through the film.
  --> From the internally reflected amount, ARA is re-absorbed, ARR is
      internally reflected, and ART is emitted through the film.
  --> Continuing this process to infinity and adding up all the outwards
      emissions gives the following:

         E_total = A + AT + ART + AR^2T + AR^3 T + ...
                 = A + AT (1 + R + R^2 + R^3 + ...)
                 = A (1 + T / (1 - R))

  Noting that E_total and A_total are equivalent, we can use this function
  for both incoming and outgoing emissions.

  Args:
    absorptivity: Balloon film's absorptivity/emissivity.
    reflectivity: Balloon film's reflectivity.

  Returns:
    total_absorptivity_factor: Factor of radiation absorbed/emitted by balloon.
  """
  transmisivity = 1.0 - absorptivity - reflectivity
  total_absorptivity_factor = absorptivity * (1.0 + transmisivity /
                                              (1.0 - reflectivity))
  if total_absorptivity_factor < 0.0 or total_absorptivity_factor > 1.0:
    raise ValueError(
        'total_absorptivity: '
        'Computed total absorptivity factor out of expected range [0, 1].')

  return total_absorptivity_factor


def convective_heat_air_factor(balloon_radius: float,
                               balloon_temperature_k: float,
                               ambient_temperature_k: float,
                               pressure_altitude_pa: float) -> float:
  """Convective heat air factor."""
  viscosity = 1.458e-6 * (ambient_temperature_k**1.5) / (
      ambient_temperature_k + 110.4)
  conductivity = 0.0241 * ((ambient_temperature_k / 273.15)**0.9)
  prandtl = 0.804 - 3.25e-4 * ambient_temperature_k
  air_density = (
      pressure_altitude_pa * _DRY_AIR_MOLAR_MASS /
      (_UNIVERSAL_GAS_CONSTANT * ambient_temperature_k))

  grashof = (9.80665 * (air_density**2) * ((2 * balloon_radius)**3) /
             (ambient_temperature_k *
              (viscosity**2))) * np.abs(ambient_temperature_k -
                                        balloon_temperature_k)
  rayleigh = prandtl * grashof
  nusselt = (2 + 0.457 * (rayleigh**0.25) +
             ((1 + 2.69e-8 * rayleigh)**(1.0 / 12.0)))
  k_heat_transfer = nusselt * conductivity / (2 * balloon_radius)

  return k_heat_transfer * (ambient_temperature_k - balloon_temperature_k)


def d_balloon_temperature_dt(balloon_volume: float,
                             balloon_mass: float,
                             balloon_temperature_k: float,
                             ambient_temperature_k: float,
                             pressure_altitude_pa: float,
                             solar_elevation_deg: float,
                             solar_flux: float,
                             earth_flux: float) -> float:
  """Compute d_balloon_temperature / dt. Assumes PE01 film.

  Args:
    balloon_volume: Balloon volume [m^3].
    balloon_mass: Balloon envelope mass [kg].
    balloon_temperature_k: Balloon temperature [K].
    ambient_temperature_k: Ambient temperature around the balloon [K].
    pressure_altitude_pa: Balloon's pressure altitude [Pa].
    solar_elevation_deg: Solar elevation [deg].
    solar_flux: Solar flux [W/m^2].
    earth_flux: Earth radiation experienced at balloon [W/m^2].

  Returns:
    d_balloon_temperature / dt: Derivative of balloon temperature [K/s].
  """

  # Compute balloon radius and surface area. Assumes spherical balloon.
  balloon_radius = (3 * balloon_volume / (4 * np.pi))**(1 / 3)
  balloon_area = 4 * np.pi * balloon_radius * balloon_radius

  # Compute atmospheric attenuation.
  atm_attenuation = solar.solar_atmospheric_attenuation(solar_elevation_deg,
                                                        pressure_altitude_pa)
  # Compute solar radiative heat.
  q_solar = (
      solar_flux * atm_attenuation * _SOLAR_VIEW_FACTOR * balloon_area *
      total_absorptivity(_PE01_ABSORPTIVITY_SOLAR, _PE01_REFLECTIVITY))

  # Compute earth radiative heat.
  q_earth = (
      earth_flux * _EARTH_VIEW_FACTOR * balloon_area * total_absorptivity(
          absorptivity_ir(black_body_flux_to_temperature(earth_flux)),
          _PE01_REFLECTIVITY))

  # Compute balloon emissions given balloon temperature.
  q_emitted = (
      black_body_temperature_to_flux(balloon_temperature_k) * balloon_area *
      total_absorptivity(absorptivity_ir(balloon_temperature_k),
                         _PE01_REFLECTIVITY))

  # Compute external convective heat given balloon temperature.
  q_convective = balloon_area * convective_heat_air_factor(
      balloon_radius, balloon_temperature_k, ambient_temperature_k,
      pressure_altitude_pa)

  # Compute derivative of balloon temperature give total heat loads and mass.
  return (q_solar + q_earth + q_convective - q_emitted) / (
      _PE01_FILM_SPECIFIC_HEAT * balloon_mass)
