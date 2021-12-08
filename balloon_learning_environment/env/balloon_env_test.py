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

"""Tests for balloon_learning_environment.env.balloon_env."""

import datetime as dt
import functools
import random
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env import balloon_arena
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.env import features
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import test_helpers
from balloon_learning_environment.utils import units
from flax.metrics import tensorboard
import gym
import jax
import numpy as np

START_DATE_TIME = units.datetime(2013, 3, 25, 9, 25, 32)


class BalloonEnvTest(parameterized.TestCase):

  def setUp(self):
    super(BalloonEnvTest, self).setUp()
    test_helpers.bind_environment_gin_parameters(seed=0)
    self.atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))
    self.create_balloon = functools.partial(
        test_helpers.create_balloon, atmosphere=self.atmosphere)

  def test_observation_space_matches_observation(self):
    env = balloon_env.BalloonEnv()
    shape = env.observation_space.sample().shape

    # Test the shape from reset
    observation = env.reset()
    self.assertEqual(observation.shape, shape)

    # Test the shape from multiple environment steps
    for _ in range(100):
      obs, _, _, _ = env.step(random.randrange(3))
      self.assertEqual(obs.shape, shape)

  def test_out_of_power(self):
    env = balloon_env.BalloonEnv()
    env.arena._balloon = self.create_balloon(
        date_time=units.datetime(2021, 9, 9, 0))  # Nighttime.
    for _ in range(10):
      env.arena._balloon.state.battery_charge = (
          env.arena._balloon.state.battery_capacity)
      _, _, is_terminal, info = env.step(random.randrange(3))
      self.assertFalse(is_terminal)
      self.assertFalse(info['out_of_power'])
    # Create an out of battery situation.
    env.arena._balloon.state.battery_charge = (
        1e-7 * env.arena._balloon.state.battery_capacity)
    _, _, is_terminal, info = env.step(random.randrange(3))
    self.assertTrue(is_terminal)
    self.assertTrue(info['out_of_power'])

  def test_time_elapsed(self):
    arena = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    time_elapsed = dt.timedelta()
    test_helpers.bind_environment_gin_parameters(arena=arena, seed=1)
    env = balloon_env.BalloonEnv()
    for _ in range(10):
      _, _, _, info = env.step(0)
      time_elapsed += constants.AGENT_TIME_STEP
      self.assertEqual(info['time_elapsed'], time_elapsed)

  @parameterized.named_parameters(
      dict(testcase_name='near_center', radius=50.0, x_km=1.0, y_km=-1.0),
      dict(testcase_name='near_border_1', radius=50.0, x_km=49.99, y_km=0.0),
      dict(testcase_name='near_border_2', radius=50.0, x_km=0.0, y_km=-49.99),
      dict(testcase_name='near_border_3', radius=50.0, x_km=-35.355, y_km=35.3),
      dict(testcase_name='10km_near_border', radius=10.0, x_km=-9.99, y_km=0.0))
  def test_reward_in_radius_should_be_one(self, radius, x_km, y_km):
    x = units.Distance(km=x_km)
    y = units.Distance(km=y_km)
    balloon_state = self.create_balloon(x, y).state
    arena = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    arena.get_balloon_state = mock.MagicMock(return_value=balloon_state)

    test_helpers.bind_environment_gin_parameters(
        seed=0,
        station_keeping_radius_km=radius,
        reward_dropoff=0.0,
        arena=arena)
    env = balloon_env.BalloonEnv()

    _, reward, _, _ = env.step(0)

    self.assertEqual(reward, 1.0)

  @parameterized.named_parameters(
      dict(testcase_name='zero_drop', radius_km=50.0, angle=0.6, dropoff=0.0),
      dict(
          testcase_name='nonzero_drop', radius_km=50.0, angle=1.3, dropoff=0.4),
      dict(testcase_name='10km_radius', radius_km=10.0, angle=2.1, dropoff=0.0))
  def test_reward_is_equal_to_dropoff_immediately_outside_radius(
      self, radius_km: float, angle: float, dropoff: float):
    # Calculate the x, y coordinates in meters just outside the radius at angle
    outside_radius_distance = units.Distance(km=radius_km + 0.1)
    x_pos = outside_radius_distance * np.cos(angle)
    y_pos = outside_radius_distance * np.sin(angle)
    balloon_state = self.create_balloon(x_pos, y_pos).state

    arena = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    arena.get_balloon_state = mock.MagicMock(return_value=balloon_state)

    test_helpers.bind_environment_gin_parameters(
        seed=0,
        station_keeping_radius_km=radius_km,
        reward_dropoff=dropoff,
        arena=arena)
    env = balloon_env.BalloonEnv()

    _, reward, _, _ = env.step(0)

    self.assertAlmostEqual(reward, dropoff, delta=0.001)

  def test_reward_is_half_after_decay_distance(self):
    # 51 km from origin, 1 km from border
    x1, y1 = units.Distance(m=47_548.69), units.Distance(m=18_442.39)
    # 101 km from origin, 51 km from border
    x2, y2 = units.Distance(m=94_165.06), units.Distance(m=36_523.16)

    balloon_state1 = self.create_balloon(x=x1, y=y1).state
    balloon_state2 = self.create_balloon(x=x2, y=y2).state

    arena1 = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    arena1.get_balloon_state = mock.MagicMock(return_value=balloon_state1)
    arena2 = balloon_arena.BalloonArena(features.PerciatelliFeatureConstructor)
    arena2.get_balloon_state = mock.MagicMock(return_value=balloon_state2)

    test_helpers.bind_environment_gin_parameters(
        seed=0,
        station_keeping_radius_km=50.0,
        reward_dropoff=1.0,
        reward_halflife=50.0,
        arena=arena1)
    env1 = balloon_env.BalloonEnv()

    test_helpers.bind_environment_gin_parameters(
        seed=0,
        station_keeping_radius_km=50.0,
        reward_dropoff=1.0,
        reward_halflife=50.0,
        arena=arena2)
    env2 = balloon_env.BalloonEnv()

    _, reward1, _, _ = env1.step(0)
    _, reward2, _, _ = env2.step(0)

    self.assertAlmostEqual(reward1 * 0.5, reward2, delta=0.001)

  @parameterized.named_parameters(
      dict(
          testcase_name='excess_energy_down',
          excess_energy=True,
          action=0,
          expected_reward=1.0),
      dict(
          testcase_name='excess_energy_stay',
          excess_energy=True,
          action=1,
          expected_reward=1.0),
      dict(
          testcase_name='no_excess_energy_down',
          excess_energy=False,
          action=0,
          expected_reward=0.95),
      dict(
          testcase_name='no_excess_energy_stay',
          excess_energy=False,
          action=1,
          expected_reward=1.0))
  def test_power_regularization_is_applied_correclty_to_reward(
      self, excess_energy: bool, action: int, expected_reward: float):
    # Mock the distance function to always return 0, so base reward is 1.0.
    with mock.patch.object(units, 'relative_distance',
                           mock.MagicMock(return_value=units.Distance(m=0.0))):
      test_helpers.bind_environment_gin_parameters(seed=0)
      env = balloon_env.BalloonEnv()
      type(env.arena.get_balloon_state()).excess_energy = mock.PropertyMock(
          return_value=excess_energy)

      _, reward, _, _ = env.step(action)

      self.assertAlmostEqual(reward, expected_reward, places=2)

  def test_seeding_gives_deterministic_initial_balloon_state(self):
    test_helpers.bind_environment_gin_parameters(seed=123)
    env1 = balloon_env.BalloonEnv()
    env2 = balloon_env.BalloonEnv()

    balloon_state1 = env1.get_simulator_state().balloon_state
    balloon_state2 = env2.get_simulator_state().balloon_state

    self.assertEqual(balloon_state1, balloon_state2)

  def test_different_seed_gives_different_initial_balloon_state(self):
    test_helpers.bind_environment_gin_parameters(seed=124)
    env1 = balloon_env.BalloonEnv()
    test_helpers.bind_environment_gin_parameters(seed=125)
    env2 = balloon_env.BalloonEnv()

    balloon_state1 = env1.get_simulator_state().balloon_state
    balloon_state2 = env2.get_simulator_state().balloon_state

    self.assertNotEqual(balloon_state1, balloon_state2)

  def test_seeding_gives_deterministic_trajectory(self):
    test_helpers.bind_environment_gin_parameters(seed=1)
    env1 = balloon_env.BalloonEnv()
    env2 = balloon_env.BalloonEnv()

    for action in (0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 0):
      env1.step(action)
      env2.step(action)

    balloon_state1 = env1.get_simulator_state().balloon_state
    balloon_state2 = env2.get_simulator_state().balloon_state

    self.assertEqual(balloon_state1, balloon_state2)

  def test_set_summary_writer(self):
    env = balloon_env.BalloonEnv()
    self.assertIsNone(env.summary_writer)
    env.set_summary_writer(None)  # None is a valid argument
    self.assertIsNone(env.summary_writer)
    env.set_summary_writer(
        tensorboard.SummaryWriter(self.create_tempdir().full_path))
    self.assertIsNotNone(env.summary_writer)

  def test_gather_summary_calls_with_defaults(self):
    env = balloon_env.BalloonEnv()
    env.summary_writer = mock.MagicMock()
    env.summary_writer.scalar = mock.MagicMock()
    env.arena.get_summaries = mock.MagicMock()
    for i in range(10):
      _, _, _, _ = env.step(i % 3)
      self.assertEqual(i, env.arena.get_summaries.call_args_list[-1][0][1])
      self.assertEqual('Balloon/Actions',
                       env.summary_writer.scalar.call_args_list[-1][0][0])
      self.assertEqual(i % 3,
                       env.summary_writer.scalar.call_args_list[-1][0][1])

  def test_gather_summary_calls_with_rendering(self):
    env = balloon_env.BalloonEnv(renderer=mock.MagicMock())
    env.summary_writer = mock.MagicMock()
    env.summary_writer.scalar = mock.MagicMock()
    env.arena.get_summaries = mock.MagicMock()
    for i in range(10):
      _, _, _, _ = env.step(i % 3)
      self.assertEqual(i, env.arena.get_summaries.call_args_list[-1][0][1])
      self.assertEqual('tensorboard',
                       env._renderer.render.call_args_list[-1][0][0])
      self.assertEqual(i, env._renderer.render.call_args_list[-1][0][2])
      self.assertEqual('Balloon/Actions',
                       env.summary_writer.scalar.call_args_list[-1][0][0])
      self.assertEqual(i % 3,
                       env.summary_writer.scalar.call_args_list[-1][0][1])


if __name__ == '__main__':
  absltest.main()
