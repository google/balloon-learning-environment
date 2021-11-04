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

"""Spherical geometry functions."""

import copyreg
import math
from typing import Any

from balloon_learning_environment.utils import units

import s2sphere as s2


# We use the spherical Earth approximation rather than WGS-84, which simplifies
# things and is appropriate for the use case of our simulator.
_EARTH_RADIUS = units.Distance(km=6371)


# Note: We _must_ register a pickle function for LatLng, otherwise
# they break when used with gin or dataclasses.astuple. Basically anywhere
# the value may be copied.
# This module should be included most places s2LatLng is used eventually,
# but it may be beneficial to enforce it is included at some later date.
# These type hints are bad, but it's what copyreg wants 😬.
def pickle_latlng(obj: Any) -> tuple:  # pylint: disable=g-bare-generic
  return s2.LatLng.from_degrees, (obj.lat().degrees, obj.lng().degrees)

copyreg.pickle(s2.LatLng, pickle_latlng)


def calculate_latlng_from_offset(center_latlng: s2.LatLng,
                                 x: units.Distance,
                                 y: units.Distance) -> s2.LatLng:
  """Calculates a new lat lng given an origin and x y offsets.

  Args:
    center_latlng: The starting latitude and longitude.
    x: An offset from center_latlng parallel to longitude.
    y: An offset from center_latlng parallel to latitude.

  Returns:
    A new latlng that is the specified distance from the start latlng.
  """
  # x and y are swapped to give heading with 0 degrees = North.
  # This is equivalent to pi / 2 - atan2(y, x).
  heading = math.atan2(x.km, y.km)  # In radians.
  angle = units.relative_distance(x, y) / _EARTH_RADIUS  # In radians.

  cos_angle = math.cos(angle)
  sin_angle = math.sin(angle)
  sin_from_lat = math.sin(center_latlng.lat().radians)
  cos_from_lat = math.cos(center_latlng.lat().radians)

  sin_lat = (cos_angle * sin_from_lat +
             sin_angle * cos_from_lat * math.cos(heading))
  d_lng = math.atan2(sin_angle * cos_from_lat * math.sin(heading),
                     cos_angle - sin_from_lat * sin_lat)

  new_lat = math.asin(sin_lat)
  new_lat = min(max(new_lat, -math.pi / 2.0), math.pi / 2.0)
  new_lng = center_latlng.lng().radians + d_lng

  return s2.LatLng.from_radians(new_lat, new_lng).normalized()
