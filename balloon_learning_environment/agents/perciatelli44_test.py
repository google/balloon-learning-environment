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

"""Tests for perciatelli44."""

from absl.testing import absltest
from balloon_learning_environment.agents import perciatelli44
import numpy as np


class Perciatelli44Test(absltest.TestCase):

  def setUp(self):
    super(Perciatelli44Test, self).setUp()
    self._perciatelli44 = perciatelli44.Perciatelli44(3, [1099])
    self._observation = np.ones(1099, dtype=np.float32)

  def test_begin_episode_returns_valid_action(self):
    action = self._perciatelli44.begin_episode(self._observation)
    self.assertIn(action, [0, 1, 2])

  def test_step_returns_valid_action(self):
    action = self._perciatelli44.begin_episode(self._observation)
    self.assertIn(action, [0, 1, 2])

  def test_perciatelli_only_accepts_3_actions(self):
    with self.assertRaises(ValueError):
      perciatelli44.Perciatelli44(4, [1099])

  def test_perciatelli_only_accepts_1099_dim_observation(self):
    with self.assertRaises(ValueError):
      perciatelli44.Perciatelli44(3, [1100])


if __name__ == '__main__':
  absltest.main()
