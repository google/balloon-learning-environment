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

"""Tests for balloon_learning_environment.metrics.console_collector."""

import os.path as osp
from unittest import mock

from absl import flags
from absl import logging
from absl.testing import absltest
from balloon_learning_environment.metrics import console_collector
from balloon_learning_environment.metrics import statistics_instance
import gin
import numpy as np
import tensorflow as tf


class ConsoleCollectorTest(absltest.TestCase):

  def _bind_gin_parameters(self, fine_grained_logging, fine_grained_frequency,
                           save_to_file):
    gin.bind_parameter('ConsoleCollector.fine_grained_logging',
                       fine_grained_logging)
    gin.bind_parameter('ConsoleCollector.fine_grained_frequency',
                       fine_grained_frequency)
    gin.bind_parameter('ConsoleCollector.save_to_file', save_to_file)

  def setUp(self):
    super().setUp()
    self._na = 5
    self._tmpdir = flags.FLAGS.test_tmpdir
    gin.clear_config()
    self._fine_grained_logging = False
    self._fine_grained_frequency = 10
    self._bind_gin_parameters(
        fine_grained_logging=self._fine_grained_logging,
        fine_grained_frequency=self._fine_grained_frequency,
        save_to_file=True)

  def test_without_gin_parameters(self):
    gin.clear_config()
    with self.assertRaises(RuntimeError):
      console_collector.ConsoleCollector(self._tmpdir, self._na, 0)

  def test_valid_creation(self):
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    self.assertEqual(collector._base_dir,
                     osp.join(self._tmpdir, 'metrics/console'))
    self.assertTrue(osp.exists(collector._base_dir))
    self.assertEqual(collector._log_file,
                     osp.join(self._tmpdir, 'metrics/console/console.log'))
    self.assertEqual(collector._fine_grained_logging,
                     self._fine_grained_logging)
    self.assertEqual(collector._fine_grained_frequency,
                     self._fine_grained_frequency)

  def test_valid_creation_no_base_dir(self):
    collector = console_collector.ConsoleCollector(None, self._na, 0)
    self.assertIsNone(collector._base_dir)
    self.assertIsNone(collector._log_file)
    self.assertEqual(collector._fine_grained_logging,
                     self._fine_grained_logging)
    self.assertEqual(collector._fine_grained_frequency,
                     self._fine_grained_frequency)

  def test_valid_creation_no_save_to_file(self):
    gin.clear_config()
    self._bind_gin_parameters(
        fine_grained_logging=self._fine_grained_logging,
        fine_grained_frequency=self._fine_grained_frequency,
        save_to_file=False)
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    self.assertEqual(collector._base_dir,
                     osp.join(self._tmpdir, 'metrics/console'))
    self.assertTrue(osp.exists(collector._base_dir))
    self.assertIsNone(collector._log_file)
    self.assertEqual(collector._fine_grained_logging,
                     self._fine_grained_logging)
    self.assertEqual(collector._fine_grained_frequency,
                     self._fine_grained_frequency)

  def test_pre_training(self):
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    self.assertIsInstance(collector._log_file_writer, tf.io.gfile.GFile)

  def test_pre_training_no_save_to_file(self):
    gin.clear_config()
    self._bind_gin_parameters(
        fine_grained_logging=self._fine_grained_logging,
        fine_grained_frequency=self._fine_grained_frequency,
        save_to_file=False)
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    with self.assertRaises(AttributeError):
      _ = collector._log_file_writer

  def test_begin_episode(self):
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    # _action_counts is not created until `pre_training` is called.
    with self.assertRaises(AttributeError):
      _ = collector._action_counts
    # _current_episode_reward is not created until `pre_training` is called.
    with self.assertRaises(AttributeError):
      _ = collector._current_episode_reward
    # _current_episode_length is not created until `pre_training` is called.
    with self.assertRaises(AttributeError):
      _ = collector._current_episode_length
    collector.begin_episode()
    self.assertTrue(
        (np.zeros(self._na) == collector._action_counts).all())
    self.assertEqual(0, collector._current_episode_reward)
    self.assertEqual(0, collector._current_episode_length)

  def test_step(self):
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    collector.begin_episode()
    expected_action_counts = np.zeros(self._na)
    num_steps = 10
    logging.info = mock.MagicMock()
    collector._log_file_writer.write = mock.MagicMock()
    for i in range(num_steps):
      action = i % self._na
      expected_action_counts[action] += 1
      collector.step(statistics_instance.StatisticsInstance(
          step=i, action=action, reward=i, terminal=False))
    # Because fine_grained_logging is off and we don't call end_episode, nothing
    # will actually get written to console/file.
    self.assertEqual(logging.info.call_count, 0)
    self.assertEqual(collector._log_file_writer.write.call_count, 0)
    self.assertTrue(
        (expected_action_counts == collector._action_counts).any())
    self.assertEqual(num_steps, collector._current_episode_length)

  def test_step_with_fine_grained_logging(self):
    gin.clear_config()
    fine_grained_logging = True
    fine_grained_frequency = 10
    self._bind_gin_parameters(fine_grained_logging=fine_grained_logging,
                              fine_grained_frequency=fine_grained_frequency,
                              save_to_file=True)
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    collector.begin_episode()
    expected_action_counts = np.zeros(self._na)
    num_steps = 100
    logging.info = mock.MagicMock()
    collector._log_file_writer.write = mock.MagicMock()
    for i in range(num_steps):
      action = i % self._na
      expected_action_counts[action] += 1
      stat_str = (
          f'Step: {i}, action: {action}, reward: {i}, terminal: False\n')
      collector.step(statistics_instance.StatisticsInstance(
          step=i, action=action, reward=i, terminal=False))
      if i % fine_grained_frequency == 0:
        logging.info.assert_called_with(stat_str)
        collector._log_file_writer.write.assert_called_with(stat_str)
    self.assertEqual(logging.info.call_count, 10)
    self.assertEqual(collector._log_file_writer.write.call_count, 10)
    self.assertTrue(
        (expected_action_counts == collector._action_counts).any())
    self.assertEqual(num_steps, collector._current_episode_length)

  def test_step_with_invalid_action(self):
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    collector.begin_episode()
    with self.assertRaises(ValueError):
      collector.step(statistics_instance.StatisticsInstance(
          step=0, action=-1, reward=0, terminal=False))
    with self.assertRaises(ValueError):
      collector.step(statistics_instance.StatisticsInstance(
          step=0, action=self._na, reward=0, terminal=False))

  def test_end_episode(self):
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    collector.begin_episode()
    logging.info = mock.MagicMock()
    collector._log_file_writer.write = mock.MagicMock()
    collector._log_file_writer.close = mock.MagicMock()
    collector.end_episode(statistics_instance.StatisticsInstance(
        step=1, action=2, reward=3, terminal=True))
    action_counts = np.zeros(self._na)
    action_counts[2] = 1
    stat_str = (
        'Episode 1 reward: 3.0, episode length: 1.0, '
        f'Action distribution: {action_counts}, '
        'Average reward: 3.0, Average episode length: 1.0\n')
    logging.info.assert_called_with(stat_str)
    collector._log_file_writer.write.assert_called_with(stat_str)
    self.assertEqual(logging.info.call_count, 1)
    self.assertEqual(collector._log_file_writer.write.call_count, 1)
    self.assertEqual(collector._log_file_writer.close.call_count, 0)
    self.assertTrue((action_counts == collector._action_counts).any())
    self.assertEqual(1, collector._current_episode_length)
    self.assertEqual(1, collector._average_episode_length)

  def test_end_training(self):
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    logging.info = mock.MagicMock()
    collector._log_file_writer.write = mock.MagicMock()
    collector._log_file_writer.close = mock.MagicMock()
    collector.end_training()
    self.assertEqual(logging.info.call_count, 0)
    self.assertEqual(collector._log_file_writer.write.call_count, 0)
    self.assertEqual(collector._log_file_writer.close.call_count, 1)
    self.assertEqual(0, collector._average_episode_reward)
    self.assertEqual(0, collector._average_episode_length)

  def test_full_run(self):
    gin.clear_config()
    fine_grained_frequency = 1
    self._bind_gin_parameters(fine_grained_logging=True,
                              fine_grained_frequency=fine_grained_frequency,
                              save_to_file=True)
    collector = console_collector.ConsoleCollector(self._tmpdir, self._na, 0)
    collector.pre_training()
    num_episodes = 2
    num_steps = 4
    for i in range(num_episodes):
      logging.info = mock.MagicMock()
      collector._log_file_writer.write = mock.MagicMock()
      collector._log_file_writer.close = mock.MagicMock()
      collector.begin_episode()
      expected_action_counts = np.zeros(self._na)
      for j in range(1, num_steps * (i + 1)):
        action = j % self._na
        expected_action_counts[action] += 1
        collector.step(statistics_instance.StatisticsInstance(
            step=j, action=action, reward=j*i, terminal=False))
      collector.end_episode(statistics_instance.StatisticsInstance(
          step=num_steps, action=num_steps, reward=num_steps, terminal=True))
      expected_action_counts[num_steps] += 1
      action_distrib = expected_action_counts / np.sum(expected_action_counts)
      episode_reward = 4.0 if i == 0 else 32.0
      episode_length = 4.0 if i == 0 else 8.0
      average_reward = 4.0 if i == 0 else 18.0
      average_episode_length = 4.0 if i == 0 else 6.0
      stat_str = (
          f'Episode {i+1} reward: {episode_reward}, '
          f'episode length: {episode_length}, '
          f'Action distribution: {action_distrib}, '
          f'Average reward: {average_reward}, '
          f'Average episode length: {average_episode_length}\n')
      logging.info.assert_called_with(stat_str)
      collector._log_file_writer.write.assert_called_with(stat_str)
      self.assertEqual(logging.info.call_count, episode_length)
      self.assertEqual(collector._log_file_writer.write.call_count,
                       episode_length)
      self.assertEqual(episode_length, collector._current_episode_length)
    collector.end_training()
    self.assertEqual(collector._log_file_writer.close.call_count, 1)
    self.assertTrue(
        (expected_action_counts == collector._action_counts).any())


if __name__ == '__main__':
  absltest.main()
