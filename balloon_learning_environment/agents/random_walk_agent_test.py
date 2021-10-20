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

"""Tests for balloon_learning_environment.agents.random_walk_agent."""

import datetime as dt
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.agents import random_walk_agent
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import sampling
from balloon_learning_environment.utils import transforms
import jax
import jax.numpy as jnp
import numpy as np


class RandomWalkAgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._num_actions = 4
    self._observation_shape = (6, 7)
    self._example_state = jnp.zeros(self._observation_shape)

  def test_agent_defaults(self):
    agent = random_walk_agent.RandomWalkAgent(self._num_actions,
                                              self._observation_shape)
    self.assertIsInstance(agent, base_agent.Agent)
    self.assertEqual(agent._time_elapsed, dt.timedelta())

  def test_sample_new_target_pressure_is_called_at_init(self):
    sampling.sample_pressure = mock.MagicMock(return_value=17.29)
    agent = random_walk_agent.RandomWalkAgent(self._num_actions,
                                              self._observation_shape,
                                              seed=1)
    rng = jax.random.PRNGKey(1)
    _, rng = jax.random.split(rng)
    sampling.sample_pressure.assert_called_once()
    mock_args, _ = sampling.sample_pressure.call_args
    self.assertLen(mock_args, 1)
    self.assertTrue((rng == mock_args[0]).all())
    self.assertEqual(17.29, agent._target_pressure)

  @parameterized.named_parameters(
      dict(
          testcase_name='above_below_threshold',
          pressure_delta=99,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='above_over_threshold',
          pressure_delta=101,
          expected_action=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='below_below_threshold',
          pressure_delta=-99,
          expected_action=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='below_over_threshold',
          pressure_delta=-101,
          expected_action=control.AltitudeControlCommand.DOWN))
  def test_select_action(self, pressure_delta, expected_action):
    agent = random_walk_agent.RandomWalkAgent(self._num_actions,
                                              self._observation_shape)
    target_pressure = constants.PERCIATELLI_PRESSURE_RANGE_MIN + 1729.0
    agent._target_pressure = target_pressure

    # TODO(joshgreaves): Use the real constructor to create feature_vector.
    feature_vector = np.zeros(1099, dtype=np.float32)
    feature_vector[0] = transforms.linear_rescale_with_saturation(
        target_pressure + pressure_delta,
        constants.PERCIATELLI_PRESSURE_RANGE_MIN,
        constants.PERCIATELLI_PRESSURE_RANGE_MAX)
    self.assertEqual(expected_action, agent._select_action(feature_vector))

  def test_begin_episode(self):
    agent = random_walk_agent.RandomWalkAgent(
        self._num_actions, self._observation_shape)
    random_walk_agent._sample_new_target_pressure = mock.MagicMock()
    mock_action = 1729
    agent._select_action = mock.MagicMock(return_value=mock_action)
    observation = np.random.rand(*self._observation_shape)
    self.assertEqual(mock_action, agent.begin_episode(observation))
    agent._select_action.assert_called_with(observation)

  def test_step(self):
    agent = random_walk_agent.RandomWalkAgent(
        self._num_actions, self._observation_shape, seed=1)
    # Overriding the targewt pressure and rng for ease of testing.
    rng = jax.random.PRNGKey(1)
    agent._rng = rng
    target_pressure = 17.29
    agent._target_pressure = target_pressure
    agent._time_elapsed = dt.timedelta(seconds=1729)
    reward = 1729.0
    observation = np.random.rand(*self._observation_shape)
    _, rng = jax.random.split(rng)
    seconds_elapsed = 1729 + 180  # 3 minutes elapsed by default.
    target_pressure += seconds_elapsed * 0.1666 * jax.random.normal(rng)
    mock_action = 7
    agent._select_action = mock.MagicMock(return_value=mock_action)
    self.assertEqual(mock_action, agent.step(reward, observation))
    agent._select_action.assert_called_with(observation)
    self.assertAlmostEqual(target_pressure, agent._target_pressure)
    self.assertEqual(agent._time_elapsed, dt.timedelta(seconds=seconds_elapsed))


if __name__ == '__main__':
  absltest.main()
