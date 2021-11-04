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

"""Tests for balloon_learning_environment.utils.sampling."""

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import sampling
from balloon_learning_environment.utils import units
import jax


class SamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Deterministic PRNG state. Tests MUST NOT rely on a specific seed.
    self.prng_key = jax.random.PRNGKey(123)
    self.atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))

  def test_sample_location_with_seed_gives_deterministic_lat_lng(self):
    location1 = sampling.sample_location(self.prng_key)
    location2 = sampling.sample_location(self.prng_key)

    self.assertEqual(location1, location2)

  def test_sample_location_gives_valid_lat_lng(self):
    latlng = sampling.sample_location(self.prng_key)

    # We only allow locations near the equator
    self.assertBetween(latlng.lat().degrees, -10.0, 10.0)
    # We don't allow locations near the international date line
    self.assertBetween(latlng.lng().degrees, -175.0, 175.0)

  def test_sample_time_with_seed_gives_deterministic_time(self):
    t1 = sampling.sample_time(self.prng_key)
    t2 = sampling.sample_time(self.prng_key)

    self.assertEqual(t1, t2)

  def test_sample_time_gives_time_within_range(self):
    # Pick a 1 hour segment to give a small valid range for testing
    begin_range = units.datetime(year=2020, month=1, day=1, hour=1)
    end_range = units.datetime(year=2020, month=1, day=1, hour=2)
    t = sampling.sample_time(
        self.prng_key, begin_range=begin_range, end_range=end_range)

    self.assertBetween(t, begin_range, end_range)

  def test_sample_pressure_with_seed_gives_deterministic_pressure(self):
    p1 = sampling.sample_pressure(self.prng_key, self.atmosphere)
    p2 = sampling.sample_pressure(self.prng_key, self.atmosphere)

    self.assertEqual(p1, p2)

  def test_sample_pressure_gives_pressure_within_range(self):
    p = sampling.sample_pressure(self.prng_key, self.atmosphere)
    self.assertBetween(p, 5000, 14000)

  def test_sample_upwelling_infrared_is_within_range(self):
    ir = sampling.sample_upwelling_infrared(self.prng_key)
    self.assertBetween(ir, 100.0, 350.0)

  @parameterized.named_parameters(
      dict(testcase_name='logit_normal', distribution_type='logit_normal'),
      dict(testcase_name='inverse_lognormal',
           distribution_type='inverse_lognormal'))
  def test_sample_upwelling_infrared_is_within_range_nondefault(
      self, distribution_type):
    ir = sampling.sample_upwelling_infrared(self.prng_key,
                                            distribution_type=distribution_type)
    self.assertBetween(ir, 100.0, 350.0)

  def test_sample_upwelling_infrared_invalid_distribution_type(self):
    with self.assertRaises(ValueError):
      sampling.sample_upwelling_infrared(self.prng_key,
                                         distribution_type='invalid')


if __name__ == '__main__':
  absltest.main()
