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

"""Helper functions for sampling locations and times for a wind field."""

import datetime as dt
from typing import Optional

from balloon_learning_environment.env.balloon import altitude_safety
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import units
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import s2sphere as s2


# Values lower than this result in an accessible pressure range that is too
# narrow.
# TODO(b/200198056): Revisit this to try and make it lower.
MIN_ALLOWABLE_UPWELLING_IR = 225.0


def sample_location(key: jnp.ndarray) -> s2.LatLng:
  """Samples a random location (lat/lng).

  Args:
    key: A key for jax random number generation.

  Returns:
    A randomly sampled location (latitude, longitude) in degrees.
  """
  lat_key, lng_key = jax.random.split(key, num=2)

  # NOTE(scandido): If we sample from the sphere we will need to seriously
  # skew the distribution (likely via rejection sampling) to retrieve enough
  # times when station keeping is possible. We temporarily (permanently?) work
  # around this by just sampling the equatorial region. Given the lower
  # latitudes have significantly lower skew we don't bother making sure each
  # lat/lng has equal chance (e.g., sample from sphere and rejection sample
  # points with latitude beyond the equatorial region). All this as
  # justification for lazy GIS work. 🤷‍♂️
  lat: float = jax.random.uniform(lat_key, (), minval=-10.0, maxval=10.0).item()
  # NOTE(scandido): We avoid sampling near the international date line so we
  # don't have to worry about wrapping the grid. This is a meteorologically
  # sound trick to save a few lines of code. 🤠
  lng: float = jax.random.uniform(
      lng_key, (), minval=-175.0, maxval=175.0).item()
  return s2.LatLng.from_degrees(lat, lng)


def sample_time(
    key: jnp.ndarray,
    begin_range: dt.datetime = units.datetime(2011, 1, 1),
    end_range: dt.datetime = units.datetime(2014, 12, 31)
) -> dt.datetime:
  """Samples a random time uniformly within the specified range.

  Args:
    key: A key for jax random number generation.
    begin_range: The earliest time that can be sampled.
    end_range: The latest time that can be sampled.

  Returns:
    A randomly sampled datetime.
  """
  time_range: dt.timedelta = end_range - begin_range
  time_offset = jax.random.choice(key, int(time_range.total_seconds()),
                                  ()).item()
  return begin_range + dt.timedelta(seconds=time_offset)


def sample_pressure(
    key: jnp.ndarray,
    atmosphere: Optional[standard_atmosphere.Atmosphere] = None) -> float:
  """Samples a pressure-level uniformly within allowable range.

  Args:
    key: A PRNGKey to use for sampling.
    atmosphere: If supplied, the atmospheric conditions will be used to decide
      on a valid max_pressure. Otherwise, a conservative value will be used.

  Returns:
    A valid pressure.
  """
  # NOTE(joshgreaves): 6493 Pa is the minimum pressure for the current
  # balloon configuration (i.e. ambient pressure when mols_air=0.0).
  # However, this could change if the balloon config/physics model is updated.
  min_pressure = 6500

  if atmosphere is not None:
    max_pressure = atmosphere.at_height(altitude_safety.MIN_ALTITUDE).pressure
  else:
    max_pressure = 11_400

  return jax.random.uniform(key,
                            minval=min_pressure,
                            maxval=max_pressure).item()


def sample_upwelling_infrared(key: jnp.ndarray,
                              distribution_type: str = 'logit_normal') -> float:
  """Sample upwelling infrared value.

  Can sample from either a LogitNormal (default) or inverse LogNormal
  distribution. Values are clipped below at 100.0.

  Args:
    key: Pseudo random number.
    distribution_type: Which distribution to use. Allowed values are
      'logit_normal' and 'inverse_lognormal'.

  Returns:
    Sampled upwelling infrared value.
  """
  # The choice of the constants used in these distributions were manually
  # selected by fitting distributions to samples from a dataset of real
  # infrared values at the top of the atmosphere.
  #
  # The probability of sampling a value below MIN_ALLOWABLE_UPWELLING_IR (225.0)
  # under the logit_normal distribution is ~0.055, so this while loop has a very
  # low likelihood of running forever.
  while True:
    if distribution_type == 'logit_normal':
      distribution = tfp.distributions.LogitNormal(2, 315)
      sample = 315 * distribution.sample(seed=key)
    elif distribution_type == 'inverse_lognormal':
      distribution = tfp.distributions.TransformedDistribution(
          tfp.distributions.LogNormal(loc=0.0, scale=1.0),
          bijector=tfp.bijectors.Chain([
              tfp.bijectors.Shift(-300.9568),
              tfp.bijectors.Scale(35),
              tfp.bijectors.Power(0.68)]))
      sample = -distribution.sample(seed=key)
    else:
      raise ValueError(f'Invalid distribution type: {distribution_type}')
    if sample >= MIN_ALLOWABLE_UPWELLING_IR:
      return sample.item()
    key, _ = jax.random.split(key, num=2)
