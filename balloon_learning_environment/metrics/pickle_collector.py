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

"""Collector class for saving episode statistics to a pickle file."""

import os.path as osp
import pickle

from balloon_learning_environment.metrics import collector
from balloon_learning_environment.metrics import statistics_instance
import tensorflow as tf


class PickleCollector(collector.Collector):
  """Collector class for reporting statistics to the console."""

  def __init__(self,
               base_dir: str,
               num_actions: int,
               current_episode: int):
    if base_dir is None:
      raise ValueError('Must specify a base directory for PickleCollector.')
    super().__init__(base_dir, num_actions, current_episode)

  def get_name(self) -> str:
    return 'pickle'

  def pre_training(self) -> None:
    pass

  def begin_episode(self) -> None:
    self._statistics = []

  def step(self, statistics: statistics_instance.StatisticsInstance) -> None:
    self._statistics.append(statistics)

  def end_episode(self,
                  statistics: statistics_instance.StatisticsInstance) -> None:
    self._statistics.append(statistics)
    pickle_file = osp.join(self._base_dir,
                           f'pickle_{self.current_episode}.pkl')
    with tf.io.gfile.GFile(pickle_file, 'w') as f:
      pickle.dump(self._statistics, f, protocol=pickle.HIGHEST_PROTOCOL)
    self.current_episode += 1

  def end_training(self) -> None:
    pass
