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

"""Tests for station_seeker_agent."""

from absl.testing import absltest
from balloon_learning_environment.agents import station_seeker_agent
import numpy as np


class StationSeekerAgentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._na = 3
    self._observation_shape = (3, 4)
    self._ss_agent = station_seeker_agent.StationSeekerAgent(
        self._na, self._observation_shape)

  def test_create_agent(self):
    self.assertEqual('StationSeekerAgent', self._ss_agent.get_name())

  def test_action_selection(self):
    mock_observation = np.zeros(1099)

    # Because the observation is uniform everywhere, we expect the controller
    # to stay (action = 1).
    for _ in range(10):  # Test for 10 episodes.
      action = self._ss_agent.begin_episode(mock_observation)
      self.assertEqual(action, 1)
      for _ in range(20):  # Each episode includes 20 steps.
        action = self._ss_agent.step(0.0, mock_observation)
        self.assertEqual(action, 1)

  def test_end_episode(self):
    # end_episode doesn't do anything (it exists to conform to the Agent
    # interface). This next line just checks that it runs without problems.
    self._ss_agent.end_episode(0.0, True)

  # TODO(bellemare): Test wind score: decreases as bearing increases.

if __name__ == '__main__':
  absltest.main()
