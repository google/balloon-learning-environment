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

"""Tests for balloon_learning_environment.metrics.collector_dispatcher."""

from absl import flags
from absl.testing import absltest
from balloon_learning_environment.metrics import collector
from balloon_learning_environment.metrics import collector_dispatcher
from balloon_learning_environment.metrics import statistics_instance


class CollectorDispatcherTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._na = 5
    self._tmpdir = flags.FLAGS.test_tmpdir

  def test_with_no_collectors(self):
    # This test verifies that we can run successfully with no collectors.
    metrics = collector_dispatcher.CollectorDispatcher(
        self._tmpdir, self._na, [], 0)
    metrics.pre_training()
    for _ in range(4):
      metrics.begin_episode()
      for _ in range(10):
        metrics.step(statistics_instance.StatisticsInstance(0, 0, 0, False))
      metrics.end_episode(
          statistics_instance.StatisticsInstance(0, 0, 0, False))
    metrics.end_training()

  def test_with_simple_collector(self):
    # Create a simple collector that keeps track of received statistics.
    logged_stats = []

    class SimpleCollector(collector.Collector):

      def get_name(self) -> str:
        return 'simple'

      def pre_training(self) -> None:
        pass

      def begin_episode(self) -> None:
        logged_stats.append([])

      def step(self, statistics) -> None:
        logged_stats[-1].append(statistics)

      def end_episode(self, statistics) -> None:
        logged_stats[-1].append(statistics)

      def end_training(self) -> None:
        pass

    # Create a simple collector that tracks method calls.
    counts = {
        'pre_training': 0,
        'begin_episode': 0,
        'step': 0,
        'end_episode': 0,
        'end_training': 0,
    }

    class CountCollector(collector.Collector):

      def get_name(self) -> str:
        return 'count'

      def pre_training(self) -> None:
        counts['pre_training'] += 1

      def begin_episode(self) -> None:
        counts['begin_episode'] += 1

      def step(self, statistics) -> None:
        counts['step'] += 1

      def end_episode(self, unused_statistics) -> None:
        counts['end_episode'] += 1

      def end_training(self) -> None:
        counts['end_training'] += 1

    # Run a collection loop.
    metrics = collector_dispatcher.CollectorDispatcher(
        self._tmpdir, self._na, [SimpleCollector, CountCollector], 0)
    metrics.pre_training()
    expected_stats = []
    num_episodes = 4
    num_steps = 10
    for _ in range(num_episodes):
      metrics.begin_episode()
      expected_stats.append([])
      for j in range(num_steps):
        stat = statistics_instance.StatisticsInstance(
            step=j, action=num_steps-j, reward=j, terminal=False)
        metrics.step(stat)
        expected_stats[-1].append(stat)
      stat = statistics_instance.StatisticsInstance(
          step=num_steps, action=0, reward=num_steps, terminal=True)
      metrics.end_episode(stat)
      expected_stats[-1].append(stat)
    metrics.end_training()
    self.assertEqual(
        counts,
        {'pre_training': 1, 'begin_episode': num_episodes,
         'step': num_episodes * num_steps, 'end_episode': num_episodes,
         'end_training': 1})
    self.assertEqual(expected_stats, logged_stats)


if __name__ == '__main__':
  absltest.main()
