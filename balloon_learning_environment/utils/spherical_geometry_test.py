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

"""Tests for spherical_geometry."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.utils import spherical_geometry
from balloon_learning_environment.utils import units

import s2sphere as s2


class SphericalGeometryTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='equator', lat=0.0, lng=45.0),
      dict(testcase_name='north_pole', lat=80.0, lng=-135.0),
      dict(testcase_name='south_pole', lat=-85.0, lng=92.0),)
  def test_offset_latlng_gives_1_degree_latitude_per_111km_everywhere(
      self, lat: float, lng: float):
    center_latlng = s2.LatLng.from_degrees(lat, lng)
    x = units.Distance(km=0.0)
    y = units.Distance(km=111.0)

    new_latlng = spherical_geometry.calculate_latlng_from_offset(
        center_latlng, x, y)

    self.assertAlmostEqual(new_latlng.lat().degrees, lat + 1.0, places=2)

  def test_offset_latlng_gives_1_degree_longitude_per_111km_at_equator(self):
    center_latlng = s2.LatLng.from_degrees(0.0, 0.0)
    x = units.Distance(km=111.0)  # About one degree longitude at equator.
    y = units.Distance(km=0.0)

    new_latlng = spherical_geometry.calculate_latlng_from_offset(
        center_latlng, x, y)

    self.assertAlmostEqual(new_latlng.lng().degrees, 1.0, places=2)

  def test_offset_latlng_gives_larger_longitude_change_away_from_equator(self):
    center_latlng = s2.LatLng.from_degrees(45.0, 0.0)
    x = units.Distance(km=111.0)  # > 1 degree longitude away from equator.
    y = units.Distance(km=0.0)

    new_latlng = spherical_geometry.calculate_latlng_from_offset(
        center_latlng, x, y)

    # The change in longitude should be greater than at the equator, but not
    # too much greater. These numbers are somewhat arbitrary - they are more
    # of a sanity check. The only alternative to this is to directly
    # re-write the formula.
    self.assertBetween(new_latlng.lng().degrees, 1.25, 1.75)

  def test_offset_latlng_wraps_around_north_pole(self):
    center_latlng = s2.LatLng.from_degrees(89.0, -90.0)
    x = units.Distance(km=0.0)
    y = units.Distance(km=222.0)  # About 2 degrees latitude.

    new_latlng = spherical_geometry.calculate_latlng_from_offset(
        center_latlng, x, y)

    # We have gone over the North pole, so latitude should be 89 again,
    # but we have gon half way around the world longitudinally.
    self.assertAlmostEqual(new_latlng.lat().degrees, 89.0, places=2)
    self.assertAlmostEqual(new_latlng.lng().degrees, 90.0, places=2)

  @parameterized.named_parameters(
      dict(
          testcase_name='west_to_east', lng=179.0, degrees=2.0,
          expected=-179.0),
      dict(
          testcase_name='east_to_west', lng=-179.0, degrees=-2.0,
          expected=179.0),
      dict(
          testcase_name='multiple_times',
          lng=0.0,
          degrees=1440,
          expected=0.0))
  def test_offset_latlng_wraps_around_longitude(self, lng: float,
                                                degrees: float,
                                                expected: float):
    center_latlng = s2.LatLng.from_degrees(0.0, lng)
    # arc_length = radius * angle_radians.
    x = spherical_geometry._EARTH_RADIUS * math.radians(degrees)
    y = units.Distance(km=0.0)

    new_latlng = spherical_geometry.calculate_latlng_from_offset(
        center_latlng, x, y)

    self.assertAlmostEqual(new_latlng.lng().degrees, expected)


if __name__ == '__main__':
  absltest.main()
