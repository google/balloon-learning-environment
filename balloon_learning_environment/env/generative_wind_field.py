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

"""A wind field created by a generative model."""

import datetime as dt
import time
from typing import List, Optional, Sequence, Union

from balloon_learning_environment.env import wind_field
from balloon_learning_environment.generative import vae
from balloon_learning_environment.models import models
from balloon_learning_environment.utils import units
import flax
import jax
from jax import numpy as jnp
import numpy as np
import scipy.interpolate


class GenerativeWindField(wind_field.WindField):
  """A wind field created by a generative model."""

  def __init__(self,
               key: Optional[jnp.ndarray] = None):
    """GenerativeWindField Constructor.

    Args:
      key: An optional key to seed the wind field with. If None, will use the
        current time in milliseconds.
    """
    super(GenerativeWindField, self).__init__()

    key = key if key else jax.random.PRNGKey(int(time.time() * 1000))

    serialized_params = models.load_offlineskies22()
    self.params = flax.serialization.msgpack_restore(serialized_params)

    self.field = None

    self.field_shape: vae.FieldShape = vae.FieldShape()
    # NOTE(scandido): We convert the field from a jax.numpy arrays to a numpy
    # arrays here, otherwise it'll be converted on the fly every time we
    # interpolate (due to scipy.interpolate). This conversion is not a huge cost
    # but the conversion on self.field (see reset() method) is significant so
    # we also do this one for completeness.
    self._grid = (
        np.array(self.field_shape.latlng_grid_points()),    # Lats.
        np.array(self.field_shape.latlng_grid_points()),    # Lngs.
        np.array(self.field_shape.pressure_grid_points()),  # Pressures.
        np.array(self.field_shape.time_grid_points()))      # Times.

  def reset_forecast(self, key: jnp.ndarray, date_time: dt.datetime) -> None:
    """Resets the wind field.

    Args:
      key: A PRNG key used to sample a new location and time for the wind field.
      date_time: An instance of a datetime object, representing the start
          of the wind field.
    """
    latents = jax.random.normal(key, shape=(64,))

    # NOTE(scandido): We convert the field from a jax.numpy array to a numpy
    # array here, otherwise it'll be converted on the fly every time we
    # interpolate (due to scipy.interpolate). This conversion is a significant
    # cost.
    decoder = vae.Decoder()
    self.field = np.array(decoder.apply(self.params, latents))

  def get_forecast(self, x: units.Distance, y: units.Distance, pressure: float,
                   elapsed_time: dt.timedelta) -> wind_field.WindVector:
    """Gets a wind in the windfield at the specified location and time.

    Args:
      x: An x offset (parallel to latitude).
      y: A y offset (parallel to longitude).
      pressure: A pressure level in pascals.
      elapsed_time: The time offset from the beginning of the wind field.

    Returns:
      The wind vector at the specified position and time.

    Raises:
      RuntimeError: if called before reset().
    """
    if self.field is None:
      raise RuntimeError('Must call reset before get_forecast.')

    point = self._prepare_get_forecast_inputs(x, y, pressure, elapsed_time)
    point = point.reshape(-1)
    uv = scipy.interpolate.interpn(
        self._grid, self.field, point, fill_value=True)
    return wind_field.WindVector(units.Velocity(mps=uv[0][0]),
                                 units.Velocity(mps=uv[0][1]))

  def get_forecast_column(
      self,
      x: units.Distance,
      y: units.Distance,
      pressures: Sequence[float],
      elapsed_time: dt.timedelta) -> List[wind_field.WindVector]:
    """A convenience function for getting multiple forecasts in a column.

    This allows a simple optimization of the generative wind field.

    Args:
      x: Distance from the station keeping target along the latitude
        parallel.
      y: Distance from the station keeping target along the longitude
        parallel.
      pressures: Multiple pressures to get a forecast for, in Pascals. (This is
        a proxy for altitude.)
      elapsed_time: Elapsed time from the "beginning" of the wind field.

    Returns:
      WindVectors for each pressure level in the WindField.

    Raises:
      RuntimeError: if called before reset().
    """
    if self.field is None:
      raise RuntimeError('Must call reset before get_forecast.')

    point = self._prepare_get_forecast_inputs(x, y, pressures, elapsed_time)
    uv = scipy.interpolate.interpn(
        self._grid, self.field, point, fill_value=True)

    result = list()
    for i in range(len(pressures)):
      result.append(wind_field.WindVector(units.Velocity(mps=uv[i][0]),
                                          units.Velocity(mps=uv[i][1])))
    return result

  @staticmethod
  def _boomerang(t: float, max_val: float) -> float:
    """Computes a value that boomerangs between 0 and max_val."""
    cycle_direction = int(t / max_val) % 2
    remainder = t % max_val

    if cycle_direction % 2 == 0:  # Forward.
      return remainder
    else:  # Backward.
      return max_val - remainder

  def _prepare_get_forecast_inputs(self,
                                   x: units.Distance,
                                   y: units.Distance,
                                   pressure: Union[Sequence[float], float],
                                   elapsed_time: dt.timedelta) -> np.ndarray:
    # TODO(bellemare): Give a units might be wrong warning if querying 10,000s
    # km away.

    # NOTE(scandido): We extend the field beyond the limits of the VAE using
    # the values at the boundary.
    x_km = x.kilometers
    y_km = y.kilometers
    x_km = np.clip(x_km, -self.field_shape.latlng_displacement_km,
                   self.field_shape.latlng_displacement_km).item()
    y_km = np.clip(y_km, -self.field_shape.latlng_displacement_km,
                   self.field_shape.latlng_displacement_km).item()
    pressure = np.clip(pressure, self.field_shape.min_pressure_pa,
                       self.field_shape.max_pressure_pa)

    # Generated wind fields have a fixed time dimension, often 48 hours.
    # Typically queries will be between 0-48 hours so most of the time it is
    # simple to query a point in the field. However, to extend the limit of
    # the field we transform times 48+ hours out to some time in the 0-48
    # well defined region in such a way that two close times will remain close
    # (and thus not have a suddenly changing wind field). We use a "boomerang",
    # i.e., time reflects backwards after 48 hours until 2*48 hours at which
    # point time goes forward and so on.
    elapsed_hours = units.timedelta_to_hours(elapsed_time)
    if elapsed_hours < self.field_shape.time_horizon_hours:
      time_field_position = elapsed_hours
    else:
      time_field_position = self._boomerang(
          elapsed_hours,
          self.field_shape.time_horizon_hours)

    num_points = 1 if isinstance(pressure, float) else len(pressure)
    point = np.empty((num_points, 4), dtype=np.float32)
    point[:, 0] = x_km
    point[:, 1] = y_km
    point[:, 2] = pressure
    point[:, 3] = time_field_position

    return point

