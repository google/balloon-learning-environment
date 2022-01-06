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

"""Tests for balloon_learning_environment.agents.marco_polo_exploration."""

import datetime as dt
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents import marco_polo_exploration
import gin
import jax
import numpy as np


class MockAgent(agent.Agent):

  def begin_episode(self, observation: np.ndarray) -> int:
    del observation
    return -1

  def step(self, reward: float, observation: np.ndarray) -> int:
    del reward
    del observation
    return -1

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    del reward
    del terminal
    pass


class MarcoPoloExplorationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._num_actions = 5
    self._observation_shape = (3, 4)
    self._probability = 0.8
    gin.bind_parameter(
        'MarcoPoloExploration.exploratory_episode_probability',
        self._probability)
    gin.bind_parameter(
        'MarcoPoloExploration.exploratory_agent_constructor', MockAgent)
    self.marco_polo = marco_polo_exploration.MarcoPoloExploration(
        self._num_actions, self._observation_shape)

  def test_defaults(self):
    self.assertEqual(self._probability,
                     self.marco_polo._exploratory_episode_probability)

  def test_begin_episode(self):
    for i in range(20):
      # We reset the RNG to test different random values.
      rng = jax.random.PRNGKey(i)
      self.marco_polo._rng = rng
      rng, _ = jax.random.split(rng)
      # At the beginning of episodes we are always in RL mode, so the same
      # action passed in will be echoed back, independent of observation.
      unused_input = np.random.rand(*self._observation_shape)
      self.assertEqual(i, self.marco_polo.begin_episode(unused_input, i))
      self.assertEqual(jax.random.uniform(rng) <= self._probability,
                       self.marco_polo._exploratory_episode)
      self.assertFalse(self.marco_polo._exploratory_phase)
      self.assertEqual(self.marco_polo._phase_time_elapsed, dt.timedelta())

  @parameterized.named_parameters(
      dict(testcase_name='exploratory_not_expired',
           exploratory_phase=True, time_elapsed=dt.timedelta(minutes=119),
           expired=False),
      dict(testcase_name='exploratory_expired',
           exploratory_phase=True, time_elapsed=dt.timedelta(minutes=120),
           expired=True),
      dict(testcase_name='rl_not_expired',
           exploratory_phase=False, time_elapsed=dt.timedelta(minutes=239),
           expired=False),
      dict(testcase_name='rl_expired',
           exploratory_phase=False, time_elapsed=dt.timedelta(minutes=240),
           expired=True))
  def test_phase_expired(self, exploratory_phase, time_elapsed, expired):
    self.marco_polo._exploratory_phase = exploratory_phase
    self.marco_polo._phase_time_elapsed = time_elapsed
    self.assertEqual(expired, self.marco_polo._phase_expired())

  @parameterized.named_parameters(
      dict(testcase_name='exploratory_not_expired',
           exploratory_episode=True, phase_expired=False),
      dict(testcase_name='exploratory_expired',
           exploratory_episode=True, phase_expired=True),
      dict(testcase_name='no_exploratory_not_expired',
           exploratory_episode=False, phase_expired=False),
      dict(testcase_name='no_exploratory_expired',
           exploratory_episode=False, phase_expired=True))
  def test_update_phase(self, exploratory_episode, phase_expired):
    self.marco_polo._exploratory_episode = exploratory_episode
    self.marco_polo._exploratory_phase = False
    self.marco_polo._phase_expired = mock.MagicMock(return_value=phase_expired)
    self.marco_polo._update_phase()
    if exploratory_episode:
      self.assertEqual(phase_expired, self.marco_polo._exploratory_phase)
      elapsed_minutes = 0 if phase_expired else 3
      self.assertEqual(dt.timedelta(minutes=elapsed_minutes),
                       self.marco_polo._phase_time_elapsed)
    else:
      self.assertFalse(self.marco_polo._exploratory_phase)
      self.assertEqual(dt.timedelta(minutes=3),
                       self.marco_polo._phase_time_elapsed)

  @parameterized.named_parameters(
      dict(testcase_name='no_exploratory', exploratory=False),
      dict(testcase_name='exploratory', exploratory=True))
  def test_step(self, exploratory):
    self.marco_polo._update_phase = mock.MagicMock()
    self.marco_polo._exploratory_phase = exploratory
    self.marco_polo._exploratory_agent.step = mock.MagicMock(return_value=-1)
    a = 3
    expected_action = -1 if exploratory else a
    unused_input = np.random.rand(*self._observation_shape)
    self.assertEqual(expected_action,
                     self.marco_polo.step(0.0, unused_input, a))
    self.marco_polo._update_phase.assert_called_once()


if __name__ == '__main__':
  absltest.main()
