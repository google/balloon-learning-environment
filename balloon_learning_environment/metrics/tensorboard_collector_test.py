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

"""Tests for balloon_learning_environment.metrics.tensorboard_collector."""

import os.path as osp
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.metrics import statistics_instance
from balloon_learning_environment.metrics import tensorboard_collector
from flax.metrics import tensorboard
import gin


class TensorboardCollectorTest(parameterized.TestCase):

  def _bind_gin_parameters(self, fine_grained_logging, fine_grained_frequency):
    gin.bind_parameter('TensorboardCollector.fine_grained_logging',
                       fine_grained_logging)
    gin.bind_parameter('TensorboardCollector.fine_grained_frequency',
                       fine_grained_frequency)

  def setUp(self):
    super().setUp()
    self._na = 5
    gin.clear_config()
    self._fine_grained_logging = True
    self._fine_grained_frequency = 10
    self._bind_gin_parameters(self._fine_grained_logging,
                              self._fine_grained_frequency)

  def test_with_invalid_base_dir_raises_value_error(self):
    with self.assertRaises(ValueError):
      tensorboard_collector.TensorboardCollector(None, self._na, 0)

  def test_without_gin_parameters_raises_runtime_error(self):
    gin.clear_config()
    with self.assertRaises(RuntimeError):
      tensorboard_collector.TensorboardCollector(
          self.create_tempdir().full_path, self._na, 0)

  def test_valid_creation_with_all_required_parameters(self):
    tensorboard.SummaryWriter = mock.MagicMock()
    base_dir = self.create_tempdir().full_path
    collector = tensorboard_collector.TensorboardCollector(
        base_dir, self._na, 0)
    self.assertEqual(collector._base_dir,
                     osp.join(base_dir, 'metrics/tensorboard'))
    self.assertTrue(osp.exists(collector._base_dir))
    self.assertEqual(collector._fine_grained_logging, True)
    self.assertEqual(collector._fine_grained_frequency,
                     self._fine_grained_frequency)
    self.assertEqual(tensorboard.SummaryWriter.call_count, 1)
    self.assertEqual(tensorboard.SummaryWriter.call_args[0][0],
                     collector._base_dir)

  def test_pre_training(self):
    collector = tensorboard_collector.TensorboardCollector(
        self.create_tempdir().full_path, self._na, 0)
    # Neither _global_step nor _num_episodes are created until `pre_training` is
    # called.
    with self.assertRaises(AttributeError):
      _ = collector._global_step
    with self.assertRaises(AttributeError):
      _ = collector._num_episodes
    collector.pre_training()
    self.assertEqual(0, collector._global_step)
    self.assertEqual(0, collector._num_episodes)

  def test_begin_episode(self):
    collector = tensorboard_collector.TensorboardCollector(
        self.create_tempdir().full_path, self._na, 0)
    # Neither _episode_length nor _episode_reward are created until
    # `begin_episode` is called.
    with self.assertRaises(AttributeError):
      _ = collector._episode_length
    with self.assertRaises(AttributeError):
      _ = collector._episode_reward
    collector.begin_episode()
    self.assertEqual(0, collector._episode_length)
    self.assertEqual(0.0, collector._episode_reward)

  @parameterized.named_parameters(
      dict(testcase_name='without_fine_logging', fine_logging=False),
      dict(testcase_name='with_fine_logging', fine_logging=True))
  def test_step(self, fine_logging):
    gin.clear_config()
    self._bind_gin_parameters(fine_logging, self._fine_grained_frequency)
    tensorboard.SummaryWriter = mock.MagicMock()
    collector = tensorboard_collector.TensorboardCollector(
        self.create_tempdir().full_path, self._na, 0)
    self.assertEqual(1, tensorboard.SummaryWriter.call_count)

    collector.summary_writer.scalar = mock.MagicMock()
    collector.summary_writer.flush = mock.MagicMock()
    collector.pre_training()
    collector.begin_episode()

    num_steps = 100
    cumulative_reward = 0.0
    for i in range(num_steps):
      action = i % self._na
      stat = statistics_instance.StatisticsInstance(
          step=i, action=action, reward=i, terminal=False)
      cumulative_reward += i
      collector.step(stat)
      if fine_logging and i % self._fine_grained_frequency == 0:
        self.assertEqual(
            'Train/FineGrainedReward',
            collector.summary_writer.scalar.call_args_list[-1][0][0])
        self.assertEqual(
            i, collector.summary_writer.scalar.call_args_list[-1][0][1])
        self.assertEqual(
            i, collector.summary_writer.scalar.call_args_list[-1][0][2])

    calls = 10 if fine_logging else 0
    self.assertEqual(collector.summary_writer.scalar.call_count, calls)
    self.assertEqual(collector.summary_writer.flush.call_count, calls)
    self.assertEqual(num_steps, collector._global_step)
    self.assertEqual(0, collector._num_episodes)
    self.assertEqual(num_steps, collector._episode_length)
    self.assertEqual(cumulative_reward, collector._episode_reward)

  @parameterized.named_parameters(
      dict(testcase_name='without_fine_logging', fine_logging=False),
      dict(testcase_name='with_fine_logging', fine_logging=True))
  def test_end_episode(self, fine_logging):
    gin.clear_config()
    self._bind_gin_parameters(fine_logging, 1)
    gin.bind_parameter('TensorboardCollector.fine_grained_logging',
                       fine_logging)
    gin.bind_parameter('TensorboardCollector.fine_grained_frequency', 1)
    tensorboard.SummaryWriter = mock.MagicMock()
    collector = tensorboard_collector.TensorboardCollector(
        self.create_tempdir().full_path, self._na, 0)
    collector.summary_writer.scalar = mock.MagicMock()
    collector.summary_writer.flush = mock.MagicMock()
    collector.pre_training()
    collector.begin_episode()
    stats = statistics_instance.StatisticsInstance(
        step=1, action=2, reward=3, terminal=True)
    collector.end_episode(stats)
    if fine_logging:
      self.assertEqual('Train/FineGrainedReward',
                       collector.summary_writer.scalar.call_args_list[-3][0][0])
      self.assertEqual(3,
                       collector.summary_writer.scalar.call_args_list[-3][0][1])
      self.assertEqual(0,
                       collector.summary_writer.scalar.call_args_list[-3][0][2])

    self.assertEqual('Train/EpisodeReward',
                     collector.summary_writer.scalar.call_args_list[-2][0][0])
    self.assertEqual(3.0,
                     collector.summary_writer.scalar.call_args_list[-2][0][1])
    self.assertEqual(0,
                     collector.summary_writer.scalar.call_args_list[-2][0][2])
    self.assertEqual('Train/EpisodeLength',
                     collector.summary_writer.scalar.call_args_list[-1][0][0])
    self.assertEqual(1,
                     collector.summary_writer.scalar.call_args_list[-1][0][1])
    self.assertEqual(0,
                     collector.summary_writer.scalar.call_args_list[-1][0][2])
    scalar_call_count = 3 if fine_logging else 2
    self.assertEqual(collector.summary_writer.scalar.call_count,
                     scalar_call_count)
    flush_call_count = 2 if fine_logging else 1
    self.assertEqual(collector.summary_writer.flush.call_count,
                     flush_call_count)
    self.assertEqual(1, collector._global_step)
    self.assertEqual(1, collector._num_episodes)
    self.assertEqual(1, collector._episode_length)
    self.assertEqual(3.0, collector._episode_reward)

  @parameterized.named_parameters(
      dict(testcase_name='without_fine_logging', fine_logging=False),
      dict(testcase_name='with_fine_logging', fine_logging=True))
  def test_full_run(self, fine_logging):
    gin.clear_config()
    self._bind_gin_parameters(fine_logging, 1)
    tensorboard.SummaryWriter = mock.MagicMock()
    collector = tensorboard_collector.TensorboardCollector(
        self.create_tempdir().full_path, self._na, 0)
    collector.summary_writer.scalar = mock.MagicMock()
    collector.summary_writer.flush = mock.MagicMock()
    collector.pre_training()
    num_episodes = 3
    global_step = 0
    for i in range(num_episodes):
      collector.begin_episode()
      num_steps = 10 * (i + 1)
      episode_reward = 0.0
      for j in range(num_steps):
        stat = statistics_instance.StatisticsInstance(
            step=j, action=j % self._na, reward=j, terminal=False)
        collector.step(stat)
        episode_reward += j
        if fine_logging:
          self.assertEqual(
              'Train/FineGrainedReward',
              collector.summary_writer.scalar.call_args_list[-1][0][0])
          self.assertEqual(
              j, collector.summary_writer.scalar.call_args_list[-1][0][1])
          self.assertEqual(
              global_step,
              collector.summary_writer.scalar.call_args_list[-1][0][2])
        global_step += 1
      stat = statistics_instance.StatisticsInstance(
          step=num_steps, action=num_steps % self._na, reward=num_steps,
          terminal=False)
      episode_reward += num_steps
      collector.end_episode(stat)
      if fine_logging:
        self.assertEqual(
            'Train/FineGrainedReward',
            collector.summary_writer.scalar.call_args_list[-3][0][0])
        self.assertEqual(
            num_steps,
            collector.summary_writer.scalar.call_args_list[-3][0][1])
        self.assertEqual(
            global_step,
            collector.summary_writer.scalar.call_args_list[-3][0][2])

      self.assertEqual('Train/EpisodeReward',
                       collector.summary_writer.scalar.call_args_list[-2][0][0])
      self.assertEqual(episode_reward,
                       collector.summary_writer.scalar.call_args_list[-2][0][1])
      self.assertEqual(i,
                       collector.summary_writer.scalar.call_args_list[-2][0][2])
      self.assertEqual('Train/EpisodeLength',
                       collector.summary_writer.scalar.call_args_list[-1][0][0])
      self.assertEqual(num_steps + 1,
                       collector.summary_writer.scalar.call_args_list[-1][0][1])
      self.assertEqual(i,
                       collector.summary_writer.scalar.call_args_list[-1][0][2])
      global_step += 1

    scalar_call_count = num_episodes * 2
    flush_call_count = num_episodes
    if fine_logging:
      scalar_call_count += global_step
      flush_call_count += global_step
    self.assertEqual(collector.summary_writer.scalar.call_count,
                     scalar_call_count)
    self.assertEqual(collector.summary_writer.flush.call_count,
                     flush_call_count)


if __name__ == '__main__':
  absltest.main()
