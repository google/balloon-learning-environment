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

"""Tests for balloon_learning_environment.agents.exploration."""

from absl.testing import absltest
from balloon_learning_environment.agents import exploration
import numpy as np


class ExplorationTest(absltest.TestCase):

  def test_exploration_class(self):
    num_actions = 5
    observation_shape = (3, 4)
    e = exploration.Exploration(num_actions, observation_shape)
    # This class just echoes back the actions passed in, ignoring all other
    # parameters.
    for i in range(5):
      self.assertEqual(
          i, e.begin_episode(np.random.rand(*observation_shape), i))
      for j in range(10):
        self.assertEqual(
            j, e.step(np.random.rand(), np.random.rand(*observation_shape), j))


if __name__ == '__main__':
  absltest.main()
