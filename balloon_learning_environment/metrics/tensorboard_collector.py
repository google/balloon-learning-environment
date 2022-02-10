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

"""Collector class for exporting statistics to Tensorboard."""

from balloon_learning_environment.metrics import collector
from balloon_learning_environment.metrics import statistics_instance
from flax.metrics import tensorboard
import gin


@gin.configurable(allowlist=['fine_grained_logging',
                             'fine_grained_frequency'])
class TensorboardCollector(collector.Collector):
  """Collector class for reporting statistics on Tensorboard."""

  def __init__(self,
               base_dir: str,
               num_actions: int,
               current_episode: int,
               fine_grained_logging: bool = False,
               fine_grained_frequency: int = 1):
    if not isinstance(base_dir, str):
      raise ValueError(
          'Must specify a base directory for TensorboardCollector.')
    super().__init__(base_dir, num_actions, current_episode)
    self._fine_grained_logging = fine_grained_logging
    self._fine_grained_frequency = fine_grained_frequency
    self.summary_writer = tensorboard.SummaryWriter(self._base_dir)

  def get_name(self) -> str:
    return 'tensorboard'

  def pre_training(self) -> None:
    # TODO(joshgreaves): This is wrong if we are starting from a checkpoint.
    self._global_step = 0

  def begin_episode(self) -> None:
    self._episode_reward = 0.0

  def _log_fine_grained_statistics(
      self, statistics: statistics_instance.StatisticsInstance) -> None:
    self.summary_writer.scalar('Train/FineGrainedReward', statistics.reward,
                               self._global_step)
    self.summary_writer.flush()

  def step(self, statistics: statistics_instance.StatisticsInstance) -> None:
    if self._fine_grained_logging:
      if self._global_step % self._fine_grained_frequency == 0:
        self._log_fine_grained_statistics(statistics)
    self._global_step += 1
    self._episode_reward += statistics.reward

  def end_episode(self,
                  statistics: statistics_instance.StatisticsInstance) -> None:
    if self._fine_grained_logging:
      self._log_fine_grained_statistics(statistics)
    self._episode_reward += statistics.reward
    self.summary_writer.scalar('Train/EpisodeReward', self._episode_reward,
                               self.current_episode)
    self.summary_writer.scalar('Train/EpisodeLength', statistics.step,
                               self.current_episode)
    self.summary_writer.flush()
    self._global_step += 1
    self.current_episode += 1

  def end_training(self) -> None:
    pass
