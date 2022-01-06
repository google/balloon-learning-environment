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

"""A model for wind noise, based on simplex noise.
"""

import dataclasses
import datetime as dt
import math

from balloon_learning_environment.utils import units
import jax
from jax import numpy as jnp
import numpy as np
import opensimplex


@dataclasses.dataclass(frozen=True)
class HarmonicParameters(object):
  """Parameters for noise harmonic."""
  weight: float
  x_spacing: float
  y_spacing: float
  pressure_spacing: float
  time_spacing: float


@dataclasses.dataclass(frozen=True)
class SimplexOffset(object):
  """Defines a displacement from the base simplex grid."""
  x: float
  y: float
  pressure: float
  time: float


# Weight, x, y, pressure, time.
_U_COMPONENT_HARMONICS = [
    HarmonicParameters(0.1445, 702.269, 2116.987, 2587.802, 245.0),
    HarmonicParameters(0.2766, 1483.570, 752.124, 646.208, 16.39),
    HarmonicParameters(0.2627, 276.810, 147.040, 587.702, 3.836),
    HarmonicParameters(0.2137, 10214.525, 1512.216, 965.629, 41.780),
    HarmonicParameters(0.1025, 181.286, 420.942, 8500.0, 245.0)
]

_V_COMPONENT_HARMONICS = [
    HarmonicParameters(0.2716, 1974.228, 2028.814, 713.697, 26.435),
    HarmonicParameters(0.2684, 699.738, 541.845, 632.116, 9.530),
    HarmonicParameters(0.2348, 217.750, 196.522, 686.825, 3.546),
    HarmonicParameters(0.1186, 47.500, 43.048, 66.553, 8.424),
    HarmonicParameters(0.1066, 3663.291, 232.023, 7499.741, 225.0)
]


# TODO(bellemare): This can be removed later. It is kRawVariance.
SIMPLEX_VARIANCE = 0.088392
OPENSIMPLEX_VARIANCE = 0.0569

# This ensures that the simplex noise variance matches the desired empirical
# variance.
# TODO(bellemare): This constant (and harmonics) need to be adjusted to match
# the 12-hour CRMSE error in Loon observations vs forecast (3.5 m/s).
# Alternatively, Brad's numbers: 4.5 m/s at 10 degree latitude.
NOISE_MAGNITUDE = math.sqrt(1.02 / OPENSIMPLEX_VARIANCE)

# TODO(bellemare): Table of parameters, per-component, per-harmonic.
# Make this a dict of dataclasses.


class NoisyWindHarmonic:
  """A sub-component of NoisyWindComponent. Used to model noise harmonics."""

  def __init__(self, params: HarmonicParameters):
    # TODO(bellemare): Might want to merge this with HarmonicParameters.
    self._simplex_generator = None
    self._offsets = None

    # TODO(bellemare): Add more harmonics from table. This is for the u
    # component, first harmonic.
    self.weight = params.weight
    self.x_spacing = params.x_spacing
    self.y_spacing = params.y_spacing
    self.pressure_spacing = params.pressure_spacing
    self.time_spacing = params.time_spacing

  def reset(self, key: jnp.ndarray) -> None:
    """Resets this harmonic's noise."""
    # We need to explicitly convert to Python int, because otherwise jax
    # will throw a tantrum inside OpenSimplex.
    wind_seed = int(jax.random.choice(key, 1634753849))

    # TODO(bellemare): Consider using non-open Simplex noise instead. In
    # particular OpenSimplex states it uses a wider kernel, so we may need
    # to at least account for this explicitly.
    self._simplex_generator = opensimplex.OpenSimplex(seed=wind_seed)
    # We generate a random translation of the simplex grid to deal with
    # the fact that OpenSimplex does not return a noise that is random at
    # the origin (it is always zero). More generally, the distribution of
    # noise at a location depends on that location.
    random_translation = jax.random.uniform(key, (4,)) * 2.0 - 1.0
    random_translation = np.asarray(random_translation)
    self._offsets = SimplexOffset(*random_translation)

  def get_noise(self, x: units.Distance, y: units.Distance, pressure: float,
                elapsed_time: dt.timedelta) -> float:
    """Returns the simplex noise for this haromnic.

    Args:
      x: Distance from the station keeping target along the latitude
        parallel.
      y: Distance from the station keeping target along the longitude
        parallel.
      pressure: Pressure at this point in the wind field in Pascals.
      elapsed_time: Elapsed time from the "beginning" of the wind field.

    Returns:
      noise: the noise value for this harmonic.

    Raises:
      ValueError: If reset() has not been called.
    """
    if self._simplex_generator is None:
      raise ValueError('Must call reset before get_noise.')

    time_in_hours = units.timedelta_to_hours(elapsed_time)

    # TODO(bellemare): One generator for all components, move out of here.
    # TODO(bellemare): What are the units for noise4d? Do we want to write
    # constants in that form?
    return NOISE_MAGNITUDE * self._simplex_generator.noise4d(
        x.km / self.x_spacing + self._offsets.x,
        y.km / self.y_spacing + self._offsets.y,
        pressure / self.pressure_spacing + self._offsets.pressure,
        time_in_hours / self.time_spacing + self._offsets.time)


class NoisyWindComponent:
  """Uses simplex noise to model one of the wind components."""

  def __init__(self, which: str):
    """Constructor for NoisyWindComponent.

    Args:
      which: Which component to construct ('u' or 'v').

    Raises:
      RuntimeError: if which is invalid.
    """
    if which == 'u':
      harmonic_params = _U_COMPONENT_HARMONICS
    elif which == 'v':
      harmonic_params = _V_COMPONENT_HARMONICS
    else:
      raise RuntimeError(f'Invalid wind component: {which}')

    # Create the harmonic helper classes based on their parameters.
    self._harmonics = [
        NoisyWindHarmonic(params) for params in harmonic_params
    ]

  def reset(self, key: jnp.ndarray) -> None:
    num_harmonics = len(self._harmonics)
    harmonic_keys = jax.random.split(key, num=num_harmonics)

    for key, harmonic in zip(harmonic_keys, self._harmonics):
      harmonic.reset(key)

  def get_noise(self, x: units.Distance, y: units.Distance, pressure: float,
                elapsed_time: dt.timedelta) -> float:
    """Returns the simplex noise at this location, for this component.

    Args:
      x: Distance from the station keeping target along the latitude
        parallel.
      y: Distance from the station keeping target along the longitude
        parallel.
      pressure: Pressure at this point in the wind field in Pascals.
      elapsed_time: Elapsed time from the "beginning" of the wind field.

    Returns:
      noise: the noise value for this component.
    """
    weighted_noise = 0.0
    total_weight = 0.0
    total_weight_squared = 0.0

    # Sum up the noise from different harmonics.
    for harmonic in self._harmonics:
      noise = harmonic.get_noise(x, y, pressure, elapsed_time)
      weighted_noise += noise * harmonic.weight
      total_weight += harmonic.weight
      total_weight_squared += harmonic.weight**2

    # This is a variance adjustement (it makes the noise larger).
    weighted_noise /= total_weight
    weighted_noise *= math.sqrt(total_weight / total_weight_squared)

    # TODO(bellemare): Perform variance correction as per BlendRandomVars.
    return weighted_noise
