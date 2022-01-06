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

"""Tests for balloon_learning_environment.agents.agent."""

from absl.testing import absltest
from balloon_learning_environment.agents import agent
import numpy as np


class AgentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._na = 5
    self._observation_shape = (3, 4)

  def test_valid_subclass(self):

    # Create a simple subclass that implements the abstract methods.
    class SimpleAgent(agent.Agent):

      def begin_episode(self, unused_obs: None) -> int:
        return 0

      def step(self, reward: float, observation: None) -> int:
        return 0

      def end_episode(self, reward: float, terminal: bool) -> None:
        pass

    simple_agent = SimpleAgent(self._na, self._observation_shape)
    self.assertEqual('SimpleAgent', simple_agent.get_name())
    self.assertEqual(self._na, simple_agent._num_actions)
    self.assertEqual(self._observation_shape, simple_agent._observation_shape)
    self.assertEqual(simple_agent.reload_latest_checkpoint(''), -1)


class RandomAgentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._na = 5
    self._observation_shape = (3, 4)
    self._observation = np.zeros(self._observation_shape, dtype=np.float32)

  def test_create_agent(self):
    random_agent = agent.RandomAgent(self._na, self._observation_shape)
    self.assertEqual('RandomAgent', random_agent.get_name())
    self.assertEqual(self._na, random_agent._num_actions)
    self.assertEqual(self._observation_shape, random_agent._observation_shape)

  def test_action_selection(self):
    random_agent = agent.RandomAgent(self._na, self._observation_shape)
    for _ in range(10):  # Test for 10 episodes.
      action = random_agent.begin_episode(self._observation)
      self.assertGreaterEqual(action, 0)
      self.assertLess(action, self._na)
      for _ in range(20):  # Each episode includes 20 steps.
        action = random_agent.step(0.0, self._observation)
        self.assertIn(action, range(self._na))
      random_agent.end_episode(0.0, True)


if __name__ == '__main__':
  absltest.main()
