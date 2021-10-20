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

"""Tests for balloon_learning_environment.utils.wind."""

from absl.testing import absltest
from balloon_learning_environment.utils import wind
import jax.numpy as jnp
import numpy as np


class WindTest(absltest.TestCase):

  def test_is_station_keeping_winds_postive_case(self):
    self.assertTrue(wind.is_station_keeping_winds(np.array([
        [1, 0], [-1, 0], [0, -1], [0, 1]])))

  def test_is_station_keeping_winds_negative_case(self):
    self.assertFalse(wind.is_station_keeping_winds(np.array([
        [1, 1.2], [2, 3.2], [10.2, 33.4]])))

  def test_wind_field_speeds_works(self):
    u = 2.1 * jnp.ones((5, 6, 7, 8))
    v = 3.2 * jnp.ones((5, 6, 7, 8))
    field = jnp.stack([u, v], axis=-1)
    self.assertAlmostEqual(wind.wind_field_speeds(field).mean().item(),
                           jnp.sqrt(2.1 * 2.1 + 3.2 * 3.2).item(),
                           places=3)
    self.assertEqual(wind.wind_field_speeds(field).shape, (5, 6, 7, 8))

  def test_mean_speed_in_wind_field_works(self):
    u = 2.1 * jnp.ones((5, 6, 7, 8))
    v = 3.2 * jnp.ones((5, 6, 7, 8))
    field = jnp.stack([u, v], axis=-1)
    self.assertAlmostEqual(wind.mean_speed_in_wind_field(field).item(),
                           jnp.sqrt(2.1 * 2.1 + 3.2 * 3.2).item(),
                           places=3)


if __name__ == '__main__':
  absltest.main()
