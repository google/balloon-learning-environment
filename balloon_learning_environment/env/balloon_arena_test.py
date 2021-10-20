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

"""Tests for balloon_learning_environment.env.balloon_arena."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env import balloon_arena
from balloon_learning_environment.env import features
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import test_helpers
from balloon_learning_environment.utils import units
import jax


class BalloonArenaTest(parameterized.TestCase):

  # TODO(joshgreaves): Patch PerciatelliFeatureConstructor, it's too slow.

  def test_int_seeding_gives_determinisic_balloon_initialization(self):
    arena1 = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    arena2 = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)

    arena1.reset(201)
    arena2.reset(201)
    balloon_state1 = arena1.get_simulator_state().balloon_state
    balloon_state2 = arena2.get_simulator_state().balloon_state

    test_helpers.compare_balloon_states(balloon_state1, balloon_state2)

  def test_array_seeding_gives_determinisic_balloon_initialization(self):
    arena1 = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    arena2 = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)

    arena1.reset(jax.random.PRNGKey(201))
    arena2.reset(jax.random.PRNGKey(201))
    balloon_state1 = arena1.get_simulator_state().balloon_state
    balloon_state2 = arena2.get_simulator_state().balloon_state
    test_helpers.compare_balloon_states(balloon_state1, balloon_state2)

  def test_different_seeds_gives_different_initialization(self):
    arena1 = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    arena2 = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)

    arena1.reset(201)
    arena2.reset(202)
    balloon_state1 = arena1.get_simulator_state().balloon_state
    balloon_state2 = arena2.get_simulator_state().balloon_state
    test_helpers.compare_balloon_states(
        balloon_state1, balloon_state2, check_not_equal=['x', 'y'])

  def test_random_seeding_doesnt_throw_exception(self):
    arena = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)

    arena.reset()
    # Succeeds if no error was thrown

  @parameterized.named_parameters((str(x), x) for x in (1, 5, 28, 90, 106, 378))
  def test_balloon_is_initialized_within_200km(self, seed: int):
    arena = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)

    arena.reset(seed)
    balloon_state = arena.get_simulator_state().balloon_state

    distance = units.relative_distance(balloon_state.x, balloon_state.y)
    self.assertLessEqual(distance.km, 200.0)

  @parameterized.named_parameters((str(x), x) for x in (1, 5, 28, 90, 106, 378))
  def test_balloon_is_initialized_within_valid_pressure_range(self, seed: int):
    arena = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)

    arena.reset(seed)
    balloon_state = arena.get_simulator_state().balloon_state

    self.assertBetween(balloon_state.pressure,
                       constants.PERCIATELLI_PRESSURE_RANGE_MIN,
                       constants.PERCIATELLI_PRESSURE_RANGE_MAX)

  def test_get_summaries(self):
    arena = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    summary_writer = mock.MagicMock()
    summary_writer.scalar = mock.MagicMock()
    summary_writer.flush = mock.MagicMock()
    arena._balloon.date_time = units.datetime(2013, 3, 25, 0, 0, 0)
    for i in range(1, 11):
      arena._balloon.state.date_time = units.datetime(2021, 6, 1, i, 0, 0)
      arena._balloon.state.battery_charge = (
          1.0 / i * arena._balloon.state.battery_capacity)
      arena._balloon.state.pressure = i * 10
      arena.get_summaries(summary_writer, i)
      self.assertEqual('Balloon/Hour',
                       summary_writer.scalar.call_args_list[-3][0][0])
      self.assertEqual(i, summary_writer.scalar.call_args_list[-3][0][1])
      self.assertEqual(i, summary_writer.scalar.call_args_list[-3][0][2])
      self.assertEqual('Balloon/NormalizedCharge',
                       summary_writer.scalar.call_args_list[-2][0][0])
      self.assertEqual(1. / i, summary_writer.scalar.call_args_list[-2][0][1])
      self.assertEqual(i, summary_writer.scalar.call_args_list[-2][0][2])
      self.assertEqual('Balloon/Pressure',
                       summary_writer.scalar.call_args_list[-1][0][0])
      self.assertEqual(i * 10, summary_writer.scalar.call_args_list[-1][0][1])
      self.assertEqual(i, summary_writer.scalar.call_args_list[-1][0][2])
    self.assertEqual(10, summary_writer.flush.call_count)


if __name__ == '__main__':
  absltest.main()
