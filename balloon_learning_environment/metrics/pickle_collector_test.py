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

"""Tests for balloon_learning_environment.metrics.pickle_collector."""

import os.path as osp
import pickle
from unittest import mock

from absl import flags
from absl.testing import absltest
from balloon_learning_environment.metrics import pickle_collector
from balloon_learning_environment.metrics import statistics_instance
import gin


class PickleCollectorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._na = 5
    self._tmpdir = flags.FLAGS.test_tmpdir
    gin.clear_config()

  def test_with_none_base_dir(self):
    with self.assertRaises(ValueError):
      pickle_collector.PickleCollector(None, self._na, 0)

  def test_valid_creation(self):
    collector = pickle_collector.PickleCollector(self._tmpdir, self._na, 0)
    self.assertEqual(collector._base_dir,
                     osp.join(self._tmpdir, 'metrics/pickle'))
    self.assertTrue(osp.exists(collector._base_dir))

  def test_pre_training(self):
    collector = pickle_collector.PickleCollector(self._tmpdir, self._na, 0)
    # _current_episode is not created until `pre_training` is called.
    with self.assertRaises(AttributeError):
      _ = collector._current_episode
    collector.pre_training()
    self.assertEqual(0, collector._current_episode)

  def test_begin_episode(self):
    collector = pickle_collector.PickleCollector(self._tmpdir, self._na, 0)
    # _statistics is not created until `begin_episode` is called.
    with self.assertRaises(AttributeError):
      _ = collector._statistics
    collector.begin_episode()
    self.assertEqual([], collector._statistics)

  def test_step(self):
    collector = pickle_collector.PickleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    collector.begin_episode()
    num_steps = 10
    expected_stats = []
    pickle.dump = mock.MagicMock()
    for i in range(num_steps):
      action = i % self._na
      stat = statistics_instance.StatisticsInstance(
          step=i, action=action, reward=i, terminal=False)
      expected_stats.append(stat)
      collector.step(stat)
    self.assertEqual(pickle.dump.call_count, 0)
    self.assertEqual(expected_stats, collector._statistics)
    self.assertEqual(0, collector._current_episode)

  def test_end_episode(self):
    collector = pickle_collector.PickleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    collector.begin_episode()
    pickle.dump = mock.MagicMock()
    expected_stats = [statistics_instance.StatisticsInstance(
        step=1, action=2, reward=3, terminal=True)]
    collector.end_episode(expected_stats[0])
    self.assertEqual(expected_stats, pickle.dump.call_args[0][0])
    self.assertEqual(pickle.dump.call_count, 1)
    self.assertEqual(expected_stats, collector._statistics)
    self.assertEqual(1, collector._current_episode)

  def test_full_run(self):
    collector = pickle_collector.PickleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    for i in range(3):
      collector.begin_episode()
      num_steps = 3 * (i + 1)
      pickle.dump = mock.MagicMock()
      expected_stats = []
      for j in range(1, num_steps):
        action = j % self._na
        stat = statistics_instance.StatisticsInstance(
            step=j, action=action, reward=j, terminal=False)
        expected_stats.append(stat)
        collector.step(stat)
      stat = statistics_instance.StatisticsInstance(
          step=num_steps, action=num_steps, reward=num_steps, terminal=True)
      expected_stats.append(stat)
      collector.end_episode(stat)
      self.assertEqual(expected_stats, pickle.dump.call_args[0][0])
      self.assertEqual(pickle.dump.call_count, 1)
      self.assertEqual(expected_stats, collector._statistics)
      self.assertEqual(i + 1, collector._current_episode)


if __name__ == '__main__':
  absltest.main()
