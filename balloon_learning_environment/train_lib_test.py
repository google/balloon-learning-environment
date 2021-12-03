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

"""Tests for balloon_learning_environment.train_lib."""

from absl import flags
from absl.testing import absltest
from balloon_learning_environment import train_lib
from balloon_learning_environment.metrics import collector
import gym
import mock


class TrainLibTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = flags.FLAGS.test_tmpdir

  # TODO(psc): Split into smaller tests.
  def test_training_loop(self):
    # This tests the full execution of the training loop, including stepping
    # through collectors.

    # An agent that always returns action 0.
    mock_agent = mock.Mock()
    mock_agent.begin_episode.return_value = 0
    mock_agent.step.return_value = 0
    mock_agent.reload_latest_checkpoint.return_value = -1

    class MockActionSpace(object):

      def __init__(self):
        self.n = 3

    # An environment that always returns a reward of 1 and returns
    # a terminal every 4 steps.
    class MockEnv(gym.Env):

      def __init__(self):
        self.action_space = MockActionSpace()
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

      def set_summary_writer(self, summary_writer):
        del summary_writer
        pass

      def render(self):
        pass

    # To test collection is happening as expected, we use a mock collector.
    steps = []
    actions = []
    rewards = []
    terminals = []
    method_counts = {
        'pre_training': 0,
        'begin_episode': 0,
        'step': 0,
        'end_episode': 0,
        'end_training': 0,
    }

    class MockCollector(collector.Collector):

      def get_name(self):
        return 'mock'

      def pre_training(self):
        method_counts['pre_training'] += 1

      def begin_episode(self):
        method_counts['begin_episode'] += 1
        steps.append([])
        actions.append([])
        rewards.append([])
        terminals.append([])

      def step(self, statistics):
        method_counts['step'] += 1
        steps[-1].append(statistics.step)
        actions[-1].append(statistics.action)
        rewards[-1].append(statistics.reward)
        terminals[-1].append(statistics.terminal)

      def end_episode(self, statistics):
        method_counts['end_episode'] += 1
        steps[-1].append(statistics.step)
        actions[-1].append(statistics.action)
        rewards[-1].append(statistics.reward)
        terminals[-1].append(statistics.terminal)

      def end_training(self):
        method_counts['end_training'] += 1

    num_episodes = 20
    max_episode_length = 10
    train_lib.run_training_loop(self.tmpdir,
                                MockEnv(),
                                mock_agent,
                                num_episodes,
                                max_episode_length,
                                [MockCollector])
    self.assertEqual(num_episodes, mock_agent.end_episode.call_count)
    # Episodes will alternate between having length 10 and length 4.
    expected_rewards = [[1.] * 5, [1.] * (max_episode_length + 1)] * 10
    self.assertEqual(expected_rewards, rewards)
    expected_actions = [[0] * 5, [0] * (max_episode_length + 1)] * 10
    self.assertEqual(expected_actions, actions)
    expected_steps = [list(range(5)), list(range(max_episode_length + 1))] * 10
    self.assertEqual(expected_steps, steps)
    for method_name, expected_count in zip(
        method_counts, [1, num_episodes, 140, num_episodes, 1]):
      self.assertEqual(method_counts[method_name], expected_count)

if __name__ == '__main__':
  absltest.main()
