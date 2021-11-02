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

"""Collector class for reporting statistics to the console."""

import os.path as osp
from typing import Union

from absl import logging
from balloon_learning_environment.metrics import collector
from balloon_learning_environment.metrics import statistics_instance
import gin
import numpy as np
import tensorflow as tf


@gin.configurable
class ConsoleCollector(collector.Collector):
  """Collector class for reporting statistics to the console."""

  def __init__(self,
               base_dir: Union[str, None],
               num_actions: int,
               current_episode: int,
               fine_grained_logging: bool = gin.REQUIRED,
               fine_grained_frequency: int = gin.REQUIRED,
               save_to_file: bool = gin.REQUIRED):
    super().__init__(base_dir, num_actions, current_episode)
    if self._base_dir is not None and save_to_file:
      self._log_file = osp.join(self._base_dir, 'console.log')
    else:
      self._log_file = None
    self._fine_grained_logging = fine_grained_logging
    self._fine_grained_frequency = fine_grained_frequency
    self._average_episode_reward = 0.0
    self._average_episode_length = 0.0

  def get_name(self) -> str:
    return 'console'

  def pre_training(self) -> None:
    if self._log_file is not None:
      self._log_file_writer = tf.io.gfile.GFile(self._log_file, 'w')

  def begin_episode(self) -> None:
    self._action_counts = np.zeros(self._num_actions)
    self._current_episode_reward = 0.0
    self._current_episode_length = 0.0

  def step(self, statistics: statistics_instance.StatisticsInstance) -> None:
    self._current_episode_reward += statistics.reward
    if statistics.action < 0 or statistics.action >= self._num_actions:
      raise ValueError(f'Invalid action: {statistics.action}')
    self._action_counts[statistics.action] += 1
    if (self._fine_grained_logging and
        self._current_episode_length % self._fine_grained_frequency == 0):
      step_string = (
          f'Step: {statistics.step}, action: {statistics.action}, '
          f'reward: {statistics.reward}, terminal: {statistics.terminal}\n')
      logging.info(step_string)
      if self._log_file is not None:
        self._log_file_writer.write(step_string)
    self._current_episode_length += 1

  def end_episode(self,
                  statistics: statistics_instance.StatisticsInstance) -> None:
    self._current_episode_reward += statistics.reward
    self._current_episode_length += 1
    self._action_counts[statistics.action] += 1
    action_distribution = self._action_counts / np.sum(self._action_counts)
    self.num_episodes += 1
    self._average_episode_reward += self._current_episode_reward
    self._average_episode_length += self._current_episode_length
    episode_string = (
        f'Episode {self.num_episodes} reward: {self._current_episode_reward}, '
        f'episode length: {self._current_episode_length}, '
        f'Action distribution: {action_distribution}, '
        f'Average reward: {self._average_episode_reward / self.num_episodes}, '
        'Average episode length: '
        f'{self._average_episode_length / self.num_episodes}\n')
    logging.info(episode_string)
    if self._log_file is not None:
      self._log_file_writer.write(episode_string)

  def end_training(self) -> None:
    if self._log_file is not None:
      self._log_file_writer.close()
