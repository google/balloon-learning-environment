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

"""Base class for metric collectors.

Each Collector should subclass this base class, as the CollectorDispatcher
object expects objects of type Collector.

The methods to implement are:
  - `get_name`: a unique identifier for subdirectory creation.
  - `pre_training`: called once before training begins.
  - `step`: called once for each training step. The parameter is an object of
    type `StatisticsInstance` which contains the statistics of the current
    training step.
  - `end_training`: called once at the end of training, and passes in a
    `StatisticsInstance` containing the statistics of the latest training step.
"""

import abc
import os.path as osp
from typing import Optional

from balloon_learning_environment.metrics import statistics_instance
import tensorflow as tf


class Collector(abc.ABC):
  """Abstract class for defining metric collectors."""

  def __init__(self,
               base_dir: Optional[str],
               num_actions: int,
               current_episode: int):
    if base_dir is not None:
      self._base_dir = osp.join(base_dir, 'metrics', self.get_name())
      # Try to create logging directory.
      try:
        tf.io.gfile.makedirs(self._base_dir)
      except tf.errors.PermissionDeniedError:
        # If it already exists, ignore exception.
        pass
    else:
      self._base_dir = None
    self._num_actions = num_actions
    self.current_episode = current_episode
    self.summary_writer = None  # Should be set by subclass, if needed.

  @abc.abstractmethod
  def get_name(self) -> str:
    pass

  @abc.abstractmethod
  def pre_training(self) -> None:
    pass

  @abc.abstractmethod
  def begin_episode(self) -> None:
    pass

  @abc.abstractmethod
  def step(self, statistics: statistics_instance.StatisticsInstance) -> None:
    pass

  @abc.abstractmethod
  def end_episode(
      self, statistics: statistics_instance.StatisticsInstance) -> None:
    pass

  @abc.abstractmethod
  def end_training(self) -> None:
    pass

  def has_summary_writer(self) -> bool:
    return self.summary_writer is not None
