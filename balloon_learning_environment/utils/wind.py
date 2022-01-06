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

"""Utility functions for evaluating a wind field."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.spatial


def is_station_keeping_winds(wind_column: np.ndarray) -> bool:
  """Determines if a wind column supports station keeping winds.

  We are looking for winds in multiple directions so the balloon can change
  altitude and head back (so to speak) towards the target. This corresponds to
  the origin sitting within the convex hull of a column of wind vectors.

  Args:
    wind_column: A column of (u, v) wind vectors.

  Returns:
    yes or no
  """

  hull = scipy.spatial.ConvexHull(wind_column)
  support = [wind_column[i, :] for i in hull.vertices]
  hull = scipy.spatial.Delaunay(support)
  return hull.find_simplex(np.zeros(2)) >= 0


@jax.jit
def wind_field_speeds(wind_field: jnp.ndarray) -> jnp.ndarray:
  """Returns the wind speed throughout the field.

  Args:
    wind_field: A 4D wind field with u, v components.

  Returns:
    A 4D array of speeds at the same grid points.
  """

  u = wind_field[:, :, :, :, 0]
  v = wind_field[:, :, :, :, 1]
  return jnp.sqrt(u * u + v * v)


@jax.jit
def mean_speed_in_wind_field(wind_field: jnp.ndarray) -> float:
  """Returns the mean wind speed throughout the field.

  Args:
    wind_field: A 4D wind field with u, v components.

  Returns:
    The mean wind speed.
  """

  return wind_field_speeds(wind_field).mean()
