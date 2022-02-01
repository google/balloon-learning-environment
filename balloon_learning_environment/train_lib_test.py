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

"""Tests for balloon_learning_environment.train_lib."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment import train_lib
from balloon_learning_environment.agents import agent
from balloon_learning_environment.metrics import collector
import gym


class _MockActionSpace(object):

  def __init__(self):
    self.n = 3


# An environment that always returns a reward of 1 and returns
# a terminal every 4 steps.
class _MockEnv(gym.Env):

  def __init__(self):
    self.action_space = _MockActionSpace()
    self._issue_terminal = False

  def reset(self):
    self._steps = 0
    # This environment will issue a terminal signal every other episode.
    self._issue_terminal = not self._issue_terminal
    return None

  def step(self, action):
    self._steps += 1
    terminal = self._steps == 4 if self._issue_terminal else False
    return (None, 1.0, terminal, {})

  def render(self):
    pass


class _MockCollector(collector.Collector):

  def __init__(self):
    self.steps = []
    self.actions = []
    self.rewards = []
    self.terminals = []
    self.method_counts = {
        'pre_training': 0,
        'begin_episode': 0,
        'step': 0,
        'end_episode': 0,
        'end_training': 0,
    }
    self.summary_writer = None

  def get_name(self):
    return 'mock'

  def pre_training(self):
    self.method_counts['pre_training'] += 1

  def begin_episode(self):
    self.method_counts['begin_episode'] += 1
    self.steps.append([])
    self.actions.append([])
    self.rewards.append([])
    self.terminals.append([])

  def step(self, statistics):
    self.method_counts['step'] += 1
    self.steps[-1].append(statistics.step)
    self.actions[-1].append(statistics.action)
    self.rewards[-1].append(statistics.reward)
    self.terminals[-1].append(statistics.terminal)

  def end_episode(self, statistics):
    self.method_counts['end_episode'] += 1
    self.steps[-1].append(statistics.step)
    self.actions[-1].append(statistics.action)
    self.rewards[-1].append(statistics.reward)
    self.terminals[-1].append(statistics.terminal)

  def end_training(self):
    self.method_counts['end_training'] += 1


class TrainLibTest(parameterized.TestCase):

  def setUp(self):
    super(TrainLibTest, self).setUp()

    # An agent that always returns action 0.
    self.mock_agent = mock.create_autospec(agent.Agent)
    self.mock_agent.begin_episode.return_value = 0
    self.mock_agent.step.return_value = 0
    self.mock_agent.reload_latest_checkpoint.return_value = -1

  # TODO(psc): Split into smaller tests.
  def test_training_loop(self):
    # This tests the full execution of the training loop, including stepping
    # through collectors.

    num_iterations = 20
    max_episode_length = 10

    # This is required to pass in the specific collector, to inspect it later.
    mock_collector = _MockCollector()
    def create_mock_collector(*unused_args):
      return mock_collector

    train_lib.run_training_loop(
        self.create_tempdir(),
        _MockEnv(),
        self.mock_agent,
        num_iterations,
        max_episode_length, [create_mock_collector],
        episodes_per_iteration=1)
    self.assertEqual(num_iterations, self.mock_agent.end_episode.call_count)
    # Episodes will alternate between having length 10 and length 4.
    expected_rewards = [[1.] * 5, [1.] * (max_episode_length + 1)] * 10
    self.assertEqual(expected_rewards, mock_collector.rewards)
    expected_actions = [[0] * 5, [0] * (max_episode_length + 1)] * 10
    self.assertEqual(expected_actions, mock_collector.actions)
    expected_steps = [list(range(5)), list(range(max_episode_length + 1))] * 10
    self.assertEqual(expected_steps, mock_collector.steps)
    for method_name, expected_count in zip(
        mock_collector.method_counts,
        [1, num_iterations, 140, num_iterations, 1]):
      self.assertEqual(mock_collector.method_counts[method_name],
                       expected_count)

  @parameterized.parameters((1), (5), (8))
  def test_episodes_per_iteration(self, episodes_per_iteration):
    num_iterations = 20
    train_lib.run_training_loop(
        self.create_tempdir(),
        _MockEnv(),
        self.mock_agent,
        num_iterations=num_iterations,
        max_episode_length=10,
        collector_constructors=[],
        episodes_per_iteration=episodes_per_iteration)

    self.assertEqual(self.mock_agent.begin_episode.call_count,
                     num_iterations * episodes_per_iteration)

if __name__ == '__main__':
  absltest.main()
