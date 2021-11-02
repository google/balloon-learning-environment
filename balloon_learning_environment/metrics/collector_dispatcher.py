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

"""Class that runs a list of Collectors for metrics reporting.

This class is what should be called from the main binary and will call each of
the specified collectors for metrics reporting.

Each metric collector is configured via gin configs/bindings (see examples in
configs/). The constructor for each desired collector should be passed in as a
list when creating this object. All of the collectors are expected to be
subclasses of the `Collector` base class (defined in `collector.py`).

Example configuration:
```
metrics = CollectorDispatcher(base_dir, num_actions, list_of_constructors)
metrics.pre_training()
for i in range(training_steps):
  ...
  metrics.step(statistics)
metrics.end_training(statistics)
```

The statistics parameter is of type `statistics_instance.StatisticsInstance`,
and contains the raw performance statistics for the current iteration. All
processing (such as averaging) will be handled by each of the individual
collectors.
"""

from typing import Callable, Optional, Sequence

from balloon_learning_environment.metrics import collector
from balloon_learning_environment.metrics import console_collector
from balloon_learning_environment.metrics import pickle_collector
from balloon_learning_environment.metrics import statistics_instance
from balloon_learning_environment.metrics import tensorboard_collector
from flax.metrics import tensorboard

BASE_CONFIG_PATH = 'balloon_learning_environment/metrics/configs'
AVAILABLE_COLLECTORS = {
    'console': console_collector.ConsoleCollector,
    'pickle': pickle_collector.PickleCollector,
    'tensorboard': tensorboard_collector.TensorboardCollector,
}


CollectorConstructorType = Callable[[str, int, int], collector.Collector]


class CollectorDispatcher(object):
  """Class for collecting and reporting Balloon Learning Environment metrics."""

  def __init__(self, base_dir: Optional[str], num_actions: int,
               collectors: Sequence[CollectorConstructorType],
               current_episode: int):
    self._collectors = [
        collector_constructor(base_dir, num_actions, current_episode)
        for collector_constructor in collectors
    ]

  def pre_training(self) -> None:
    for c in self._collectors:
      c.pre_training()

  def begin_episode(self) -> None:
    for c in self._collectors:
      c.begin_episode()

  def step(self, statistics: statistics_instance.StatisticsInstance) -> None:
    for c in self._collectors:
      c.step(statistics)

  def end_episode(self,
                  statistics: statistics_instance.StatisticsInstance) -> None:
    for c in self._collectors:
      c.end_episode(statistics)

  def end_training(self) -> None:
    for c in self._collectors:
      c.end_training()

  def get_summary_writer(self) -> Optional[tensorboard.SummaryWriter]:
    """Returns the first found instance of a summary_writer, or None."""
    for c in self._collectors:
      if c.has_summary_writer():
        return c.summary_writer
    return None
