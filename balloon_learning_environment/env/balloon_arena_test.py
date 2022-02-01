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

"""Tests for balloon_learning_environment.env.balloon_arena."""

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

if __name__ == '__main__':
  absltest.main()
