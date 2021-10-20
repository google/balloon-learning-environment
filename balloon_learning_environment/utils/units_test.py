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

"""Tests for balloon_learning_environment.utils.units."""

import datetime as dt

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.utils import units
import numpy as np


# Disabling linter to explicitly test comparison operators operators.
# pylint: disable=g-generic-assert
class DistanceTest(parameterized.TestCase):

  def test_distance_adds_all_distances_in_constructor(self):
    d = units.Distance(m=1000.0, meters=1100.0, km=1.2, kilometers=1.3)

    self.assertEqual(d.m, 4_600.0)

  def test_distance_converts_meters_to_kilometers(self):
    d = units.Distance(m=1525.0)

    self.assertEqual(d.km, 1.525)
    self.assertEqual(d.kilometers, 1.525)

  def test_distance_converts_kilometers_to_meters(self):
    d = units.Distance(km=3.876)

    self.assertEqual(d.m, 3_876)
    self.assertEqual(d.meters, 3_876)

  def test_distance_converts_between_meters_and_feet(self):
    d = units.Distance(feet=1.0)
    d2 = units.Distance(meters=d.meters)

    self.assertEqual(d.meters, 0.3048)
    self.assertEqual(d2.feet, 1.0)

  def test_distance_divided_by_time_gives_velocity(self):
    d = units.Distance(m=10.0)
    t = dt.timedelta(seconds=5.0)

    vel = d / t
    self.assertIsInstance(vel, units.Velocity)
    self.assertEqual(vel.mps, 2.0)

  def test_distance_divided_by_float_gives_distance(self):
    d = units.Distance(m=10.0)

    d = d / 2.0
    self.assertIsInstance(d, units.Distance)
    self.assertEqual(d.m, 5.0)

  def test_distance_divided_by_distance_gives_float(self):
    d1 = units.Distance(m=10.0)
    d2 = units.Distance(m=5.0)

    result = d1 / d2

    self.assertIsInstance(result, float)
    self.assertEqual(result, 2.0)

  @parameterized.named_parameters(
      dict(testcase_name='lhs', lhs=True), dict(testcase_name='rhs', lhs=False))
  def test_distance_multiplied_by_float_gives_distance(self, lhs):
    d = units.Distance(m=2.5)

    if lhs:
      d = d * 4
    else:
      d = 4 * d

    self.assertEqual(d.m, 10.0)

  def test_distances_are_equal(self):
    d1 = units.Distance(km=1.5)
    d2 = units.Distance(m=1_500.0)

    self.assertEqual(d1, d2)

  def test_distance_are_not_equal(self):
    d1 = units.Distance(m=1.1)
    d2 = units.Distance(m=1.10001)

    self.assertNotEqual(d1, d2)

  def test_distance_less_than(self):
    d1 = units.Distance(m=1.0)
    d2 = units.Distance(m=1.0)
    d3 = units.Distance(m=1.1)

    self.assertFalse(d1 < d2)
    self.assertTrue(d1 < d3)

  def test_distance_less_than_or_equal(self):
    d1 = units.Distance(m=1.0)
    d2 = units.Distance(m=1.0)
    d3 = units.Distance(m=1.1)

    self.assertTrue(d1 <= d2)
    self.assertTrue(d1 <= d3)
    self.assertFalse(d3 <= d1)

  def test_distance_greater_than(self):
    d1 = units.Distance(m=1.0)
    d2 = units.Distance(m=1.0)
    d3 = units.Distance(m=1.1)

    self.assertFalse(d1 > d2)
    self.assertTrue(d3 > d1)

  def test_distance_greater_than_or_equal(self):
    d1 = units.Distance(m=1.0)
    d2 = units.Distance(m=1.0)
    d3 = units.Distance(m=1.1)

    self.assertTrue(d1 >= d2)
    self.assertTrue(d3 >= d1)
    self.assertFalse(d1 >= d3)


class VelocityTest(parameterized.TestCase):

  def test_velocity_adds_all_velocities_in_constructor(self):
    v = units.Velocity(
        mps=3, meters_per_second=4, kmph=3.6, kilometers_per_hour=7.2)

    self.assertEqual(v.mps, 10.0)

  def test_velocity_converts_mps_to_kmph(self):
    v = units.Velocity(mps=1.0)

    self.assertEqual(v.kmph, 3.6)
    self.assertEqual(v.kilometers_per_hour, 3.6)

  def test_velocity_converts_kmph_to_mps(self):
    v = units.Velocity(kmph=3.6)

    self.assertEqual(v.mps, 1.0)
    self.assertEqual(v.meters_per_second, 1.0)

  def test_velocities_add(self):
    v1 = units.Velocity(mps=3.5)
    v2 = units.Velocity(mps=9.3)

    v = v1 + v2
    self.assertEqual(v.mps, 3.5 + 9.3)

  def test_velocities_subtract(self):
    v1 = units.Velocity(mps=3.5)
    v2 = units.Velocity(mps=9.3)

    v = v1 - v2
    self.assertEqual(v.mps, 3.5 - 9.3)

  def test_equality_of_velocities(self):
    v1 = units.Velocity(mps=2.5)
    v2 = units.Velocity(mps=2.5)
    v3 = units.Velocity(mps=2.51)

    self.assertTrue(v1 == v2)
    self.assertFalse(v1 == v3)

  @parameterized.named_parameters(
      dict(testcase_name='lhs', lhs=True), dict(testcase_name='rhs', lhs=False))
  def test_velocity_multiplies_with_timedelta(self, lhs: bool):
    v = units.Velocity(mps=10.0)
    change_in_time = dt.timedelta(seconds=20.0)

    if lhs:
      d = v * change_in_time
    else:
      d = change_in_time * v

    self.assertIsInstance(d, units.Distance)
    self.assertAlmostEqual(d.m, 200.0)


class EnergyTest(absltest.TestCase):

  def test_energies_add(self):
    e1 = units.Energy(watt_hours=1.5)
    e2 = units.Energy(watt_hours=4.5)

    self.assertEqual((e1 + e2).watt_hours, 6.0)

  def test_energies_subtract(self):
    e1 = units.Energy(watt_hours=4.0)
    e2 = units.Energy(watt_hours=1.5)

    self.assertEqual((e1 - e2).watt_hours, 2.5)

  def test_energies_divide(self):
    e1 = units.Energy(watt_hours=9.0)
    e2 = units.Energy(watt_hours=3.0)

    # Dividing energies gives a ratio, which doesn't have a unit.
    self.assertEqual(e1 / e2, 3.0)

  def test_energies_multiply_with_float(self):
    e = units.Energy(watt_hours=2.0)
    c = 5.0

    # We should be able to left- and right-multiply.
    e = e * c
    e = c * e

    self.assertEqual(e.watt_hours, 50.0)

  def test_energy_greater_than(self):
    e1 = units.Energy(watt_hours=5.0)
    e2 = units.Energy(watt_hours=5.5)
    e3 = units.Energy(watt_hours=6.0)

    self.assertTrue(e2 > e1)
    self.assertFalse(e1 > e3)

  def test_energy_greater_than_or_equal_to(self):
    e1 = units.Energy(watt_hours=5.0)
    e2 = units.Energy(watt_hours=5.0)
    e3 = units.Energy(watt_hours=5.01)

    self.assertTrue(e2 >= e1)
    self.assertTrue(e3 >= e1)
    self.assertFalse(e1 >= e3)

  def test_equality_of_energies(self):
    e1 = units.Energy(watt_hours=3.0)
    e2 = units.Energy(watt_hours=3.0)
    e3 = units.Energy(watt_hours=3.001)

    self.assertTrue(e1 == e2)
    self.assertFalse(e1 == e3)


class PowerTest(absltest.TestCase):

  def test_powers_add(self):
    p1 = units.Power(watts=3.5)
    p2 = units.Power(watts=4.5)

    self.assertEqual((p1 + p2).watts, 8.0)

  def test_powers_subtract(self):
    p1 = units.Power(watts=5.5)
    p2 = units.Power(watts=3.0)

    self.assertEqual((p1 - p2).watts, 2.5)

  def test_powers_multiply_with_time(self):
    # Energy = power * time
    p = units.Power(watts=5.0)
    delta_time = dt.timedelta(hours=2.0)

    energy = p * delta_time

    self.assertIsInstance(energy, units.Energy)
    self.assertEqual(energy.watt_hours, 10.0)

  def test_power_greather_than(self):
    p1 = units.Power(watts=5.0)
    p2 = units.Power(watts=5.5)
    p3 = units.Power(watts=6.0)

    self.assertTrue(p2 > p1)
    self.assertFalse(p1 > p3)

  def test_equality_of_powers(self):
    p1 = units.Power(watts=3.0)
    p2 = units.Power(watts=3.0)
    p3 = units.Power(watts=3.001)

    self.assertTrue(p1 == p2)
    self.assertFalse(p1 == p3)

# pylint: enable=g-generic-assert


class UnitsTest(absltest.TestCase):

  def test_111_km_should_be_one_degree(self):
    degrees = units.distance_to_degrees(units.Distance(km=111.0))

    self.assertEqual(degrees, 1.0)

  def test_relative_distance(self):
    # Distance objects consider very close distances as equal.
    self.assertEqual(
        units.relative_distance(units.Distance(m=1.0), units.Distance(m=1.0)),
        units.Distance(m=np.sqrt(2)))

  def test_seconds_to_hours(self):
    self.assertEqual(units.seconds_to_hours(7200.0), 2.0)

  def test_timedelta_to_hours(self):
    hours = 6.39
    delta = dt.timedelta(hours=hours)
    self.assertEqual(units.timedelta_to_hours(delta), hours)


if __name__ == '__main__':
  absltest.main()
