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

"""Tests for balloon_learning_environment.agents.dqn_agent."""

import os
import pickle
from unittest import mock

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.agents import agent_registry
from balloon_learning_environment.agents import dqn_agent
from dopamine.jax.agents.dqn import dqn_agent as base_dqn_agent
import gin
import jax.numpy as jnp
import tensorflow as tf

FLAGS = flags.FLAGS


class DQNAgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_subdir = self.create_tempdir()
    self._num_actions = 4
    self._observation_shape = (6, 7)
    self._example_state = jnp.zeros(self._observation_shape)
    gin.parse_config_file(agent_registry.REGISTRY['dqn'][1])
    # Override 8 layers to 1 layer for speed ⚡️
    gin.bind_parameter('networks.MLPNetwork.num_layers', 1)

  def test_agent_defaults(self):
    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    self.assertIsInstance(agent, base_dqn_agent.JaxDQNAgent)

  def test_begin_episode(self):
    agent = dqn_agent.DQNAgent(self._num_actions,
                               self._observation_shape, seed=0)
    # An all-zeros state will produce all-zeros Q values, which will result in
    # action 0 selected by the argmax.
    action = agent.begin_episode(jnp.zeros_like(self._example_state))
    self.assertEqual(0, action)
    # Because we are using a fixed seed we can deterministically guarantee that
    # a state of all ones will pick action 1.
    action = agent.begin_episode(jnp.ones_like(self._example_state))
    self.assertEqual(1, action)

  def test_step(self):
    # To avoid random action selection.
    gin.bind_parameter('JaxDQNAgent.eval_mode', True)
    gin.bind_parameter('JaxDQNAgent.epsilon_eval', 0.0)
    agent = dqn_agent.DQNAgent(self._num_actions,
                               self._observation_shape,
                               seed=0)
    _ = agent.begin_episode(jnp.zeros_like(self._example_state))
    # An all-zeros state will produce all-zeros Q values, which will result in
    # action 0 selected by the argmax.
    action = agent.step(0.0, jnp.zeros_like(self._example_state))
    self.assertEqual(0, action)
    # Because we are using a fixed seed we can deterministically guarantee that
    # a state of all ones will pick action 1.
    action = agent.step(0., jnp.ones_like(self._example_state))
    self.assertEqual(1, action)

  def test_end_episode(self):
    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    base_dqn_agent.JaxDQNAgent.end_episode = mock.MagicMock()
    agent.end_episode(1729.0, True)
    self.assertEqual(
        1729.0, base_dqn_agent.JaxDQNAgent.end_episode.call_args_list[0][0][1])
    self.assertEqual(
        True, base_dqn_agent.JaxDQNAgent.end_episode.call_args_list[0][0][2])

  def test_set_mode_to_train_correctly_sets_internal_eval_mode_flag(self):
    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    agent.set_mode(base_agent.AgentMode.TRAIN)

    self.assertFalse(agent.eval_mode)

  def test_set_mode_to_eval_correctly_sets_internal_eval_mode_flag(self):
    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    agent.set_mode(base_agent.AgentMode.EVAL)

    self.assertTrue(agent.eval_mode)

  @parameterized.named_parameters(
      dict(testcase_name='no_bundle', bundle=None),
      dict(testcase_name='with_bundle', bundle={'foo': 'test'}))
  @mock.patch.object(pickle, 'dump', autospec=True)
  @mock.patch.object(tf.io.gfile, 'GFile', autospec=True)
  @mock.patch.object(logging, 'warning', autospec=True)
  def test_save_checkpoint(self, mock_logger, mock_gfile, mock_pickle, bundle):
    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    checkpoint_dir = '/tmp/test'
    iteration_number = 5
    base_dqn_agent.JaxDQNAgent.bundle_and_checkpoint = mock.MagicMock(
        return_value=bundle)
    agent.save_checkpoint(checkpoint_dir, iteration_number)
    logging_calls = 1 if bundle is None else 0
    self.assertEqual(mock_logger.call_count, logging_calls)
    if bundle is None:
      return

    self.assertEqual(mock_gfile.call_count, 1)
    self.assertEqual(
        mock_gfile.call_args[0][0],
        f'{checkpoint_dir}/checkpoint_{iteration_number:05d}.pkl')
    self.assertEqual(mock_gfile.call_args[0][1], 'w')
    self.assertEqual(mock_pickle.call_count, 1)
    self.assertEqual(mock_pickle.call_args[0][0], bundle)

  @parameterized.named_parameters(
      dict(testcase_name='no_checkpoint_no_unbundle',
           checkpoint_exists=False, unbundle=False),
      dict(testcase_name='no_checkpoint_unbundle',
           checkpoint_exists=False, unbundle=True),
      dict(testcase_name='checkpoint_no_unbundle',
           checkpoint_exists=True, unbundle=False),
      dict(testcase_name='checkpoint_unbundle',
           checkpoint_exists=True, unbundle=True))
  @mock.patch.object(pickle, 'load', autospec=True)
  @mock.patch.object(tf.io.gfile, 'GFile', autospec=True)
  @mock.patch.object(tf.io.gfile, 'exists', autospec=True)
  @mock.patch.object(logging, 'warning', autospec=True)
  def test_load_checkpoint(self, mock_logger, mock_exists, mock_gfile,
                           mock_pickle, checkpoint_exists, unbundle):
    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    checkpoint_dir = '/tmp/test'
    iteration_number = 5
    mock_exists.return_value = checkpoint_exists
    bundle = {'foo': 'test'}
    mock_pickle.return_value = bundle
    base_dqn_agent.JaxDQNAgent.unbundle = mock.MagicMock(
        return_value=unbundle)
    agent.load_checkpoint(checkpoint_dir, iteration_number)

    self.assertEqual(mock_exists.call_count, 1)
    logger_count = 1 if (not checkpoint_exists or not unbundle) else 0
    self.assertEqual(mock_logger.call_count, logger_count)
    if not checkpoint_exists:
      return

    self.assertEqual(mock_gfile.call_count, 1)
    self.assertEqual(
        mock_gfile.call_args[0][0],
        f'{checkpoint_dir}/checkpoint_{iteration_number:05d}.pkl')
    self.assertEqual(mock_gfile.call_args[0][1], 'rb')
    self.assertEqual(mock_pickle.call_count, 1)
    base_dqn_agent.JaxDQNAgent.unbundle.assert_called_once_with(
        mock.ANY, checkpoint_dir, iteration_number, bundle)

  def test_reload_latest_checkpoint_with_invalid_dir(self):
    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    self.assertEqual(
        -1, agent.reload_latest_checkpoint('/does/not/exist'))

  def test_reload_latest_checkpoint_with_empty_dir(self):
    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    self.assertEqual(
        -1, agent.reload_latest_checkpoint(self._test_subdir))

  def test_reload_latest_checkpoint(self):
    filename = os.path.join(self._test_subdir, 'checkpoint_00123.pkl')
    base_dqn_agent.JaxDQNAgent.unbundle = mock.MagicMock(
        return_value=True)
    with tf.io.gfile.GFile(filename, 'w') as f:
      pickle.dump({'data': 1}, f)

    agent = dqn_agent.DQNAgent(self._num_actions, self._observation_shape)
    self.assertEqual(123, agent.reload_latest_checkpoint(self._test_subdir))


if __name__ == '__main__':
  absltest.main()
