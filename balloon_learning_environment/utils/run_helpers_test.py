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

"""Tests for run_helpers."""

from unittest import mock

from absl.testing import absltest
from balloon_learning_environment.agents import agent
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.utils import run_helpers
from balloon_learning_environment.utils import test_helpers
import gin
import gym

_DQN_GIN_FILE = 'balloon_learning_environment/agents/configs/dqn.gin'


class RunHelpersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.gin_file = 'gin_file'

  def test_unrecognized_agent(self):
    # An unrecognized agent will raise an error.
    with self.assertRaises(ValueError):
      _ = run_helpers.get_agent_gin_file('invalid', None)
    self.assertEqual(
        run_helpers.get_agent_gin_file('invalid', self.gin_file), self.gin_file)

  def test_valid_agent_with_no_default_gin_file(self):
    # Passing a value for gin_file will return the same value, even for an
    # invalid agent.
    # The random agent has no gin file.
    self.assertIsNone(run_helpers.get_agent_gin_file('random', None))
    # When passing in a value as the second parameter it will return this.
    self.assertEqual(
        run_helpers.get_agent_gin_file('random', self.gin_file), self.gin_file)

  def test_valid_agent_with_default_gin_file(self):
    # The dqn agent has a gin file.
    self.assertIsNotNone(run_helpers.get_agent_gin_file('dqn', None))
    # When passing in a value as the second parameter it will return this
    # instead of the default.
    self.assertEqual(
        run_helpers.get_agent_gin_file('dqn', self.gin_file), self.gin_file)

  def test_create_agent_with_invalid_agent(self):
    # An unrecognized agent will raise an error.
    with self.assertRaises(ValueError):
      _ = run_helpers.create_agent('invalid', 4, (6, 7))

  def test_create_agent_with_valid_agent(self):
    # The random agent is a valid agent.
    self.assertIsInstance(
        run_helpers.create_agent('random', 4, (6, 7)), agent.Agent)

  @mock.patch.object(gin, 'parse_config_files_and_bindings', autospec=True)
  def test_bind_gin_variables_binds_agent_gin_file_correctly(
      self, mock_parse_function):
    run_helpers.bind_gin_variables('dqn')

    mock_parse_function.assert_called_with([_DQN_GIN_FILE],
                                           bindings=(),
                                           skip_unknown=False)

  @mock.patch.object(gin, 'parse_config_files_and_bindings', autospec=True)
  def test_bind_gin_variables_overrides_agent_gin_file_correctly(
      self, mock_parse_function):
    run_helpers.bind_gin_variables('dqn', agent_gin_file=self.gin_file)

    mock_parse_function.assert_called_with([self.gin_file],
                                           bindings=(),
                                           skip_unknown=False)

  @mock.patch.object(gin, 'parse_config_files_and_bindings', autospec=True)
  def test_bind_gin_variables_adds_gin_bindings_correctly(
      self, mock_parse_function):
    fake_bindings = ['binding1', 'binding2']
    run_helpers.bind_gin_variables('random', gin_bindings=fake_bindings)

    mock_parse_function.assert_called_with([],
                                           bindings=fake_bindings,
                                           skip_unknown=False)

  @mock.patch.object(gin, 'parse_config_files_and_bindings', autospec=True)
  def test_bind_gin_variables_correctly_combines_all_gin_files(
      self, mock_parse_function):
    extra_gin_file = 'extra_gin_file'
    run_helpers.bind_gin_variables(
        'dqn',
        additional_gin_files=[extra_gin_file])

    mock_parse_function.assert_called_with(
        [_DQN_GIN_FILE, extra_gin_file],
        bindings=(),
        skip_unknown=False)


if __name__ == '__main__':
  absltest.main()
