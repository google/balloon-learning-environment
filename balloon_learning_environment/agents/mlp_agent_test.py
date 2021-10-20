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

"""Tests for balloon_learning_environment.agents.mlp_agent."""

from absl.testing import absltest
from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.agents import agent_registry
from balloon_learning_environment.agents import mlp_agent
from balloon_learning_environment.agents import networks
import flax
import gin
import jax
import jax.numpy as jnp


class MLPAgentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._num_actions = 4
    self._observation_shape = (6, 7)
    self._example_state = jnp.zeros(self._observation_shape)
    gin.parse_config_file(agent_registry.REGISTRY['mlp'][1])

  def _create_network(self):
    self._network_def = networks.MLPNetwork(num_actions=self._num_actions)

  def test_select_action(self):
    self._create_network()
    network_params = self._network_def.init(jax.random.PRNGKey(0),
                                            self._example_state)
    # A state of all zeros will produce all-zero Q-values, which will result in
    # the argmax always selecting action 0.
    zeros_state = jnp.zeros_like(self._example_state)
    self.assertEqual(0, mlp_agent.select_action(self._network_def,
                                                network_params,
                                                zeros_state))
    # Because we are using a fixed seed we can deterministically guarantee that
    # a state of all ones will pick action 2.
    ones_state = jnp.ones_like(self._example_state)
    self.assertEqual(2, mlp_agent.select_action(self._network_def,
                                                network_params,
                                                ones_state))

  def test_create_optimizer(self):
    optim = mlp_agent.create_optimizer()
    self.assertIsInstance(optim, flax.optim.sgd.GradientDescent)
    self.assertEqual(0.001, optim.hyper_params.learning_rate)
    lr = 0.5
    gin.bind_parameter('mlp_agent.create_optimizer.learning_rate', lr)
    optim = mlp_agent.create_optimizer()
    self.assertEqual(lr, optim.hyper_params.learning_rate)

  def test_train(self):
    self._create_network()
    network_params = self._network_def.init(jax.random.PRNGKey(0),
                                            self._example_state)
    optim_def = mlp_agent.create_optimizer()
    optim = optim_def.create(network_params)
    # An all-zeros state will produce all-zeros Q values.
    state = jnp.zeros_like(self._example_state)
    # An all-ones next_state will produce non-zero Q-values, leading to non-zero
    # temporal difference and a non-zero gradient. This will in turn change the
    # optimizer's target.
    next_state = jnp.ones_like(self._example_state)
    loss, new_optim = mlp_agent.train(self._network_def,
                                      optim,
                                      state,
                                      0,   # action
                                      0.,  # reward
                                      next_state,
                                      0,  # next action
                                      0.9)  # gamma
    self.assertGreater(loss, 0.)  # Non-zero loss.
    # Optimizer target will have changed.
    self.assertFalse(jnp.array_equal(
        optim.target['params']['Dense_0']['kernel'],
        new_optim.target['params']['Dense_0']['kernel']))

  def test_agent_defaults(self):
    agent = mlp_agent.MLPAgent(self._num_actions, self._observation_shape)
    self.assertEqual(0.9, agent._gamma)
    # Create an optimizer with a fixed seed.
    self._create_network()
    network_params = self._network_def.init(jax.random.PRNGKey(0),
                                            self._example_state)
    optim_def = mlp_agent.create_optimizer()
    optim = optim_def.create(network_params)
    # Because we did not specify a seed to the agent, it will use one based on
    # time, which will not match the optimizer we created with a fixed seed.
    self.assertFalse(jnp.array_equal(
        optim.target['params']['Dense_0']['kernel'],
        agent.optimizer.target['params']['Dense_0']['kernel']))

  def test_agent_network(self):
    agent = mlp_agent.MLPAgent(self._num_actions, self._observation_shape,
                               gamma=0.99, seed=0)
    self._create_network()
    network_params = self._network_def.init(jax.random.PRNGKey(0),
                                            self._example_state)
    optim_def = mlp_agent.create_optimizer()
    optim = optim_def.create(network_params)
    # Because we specified a seed to the agent, it will match the one we created
    # here.
    self.assertTrue(jnp.array_equal(
        optim.target['params']['Dense_0']['kernel'],
        agent.network['params']['Dense_0']['kernel']))

  def test_begin_episode(self):
    agent = mlp_agent.MLPAgent(self._num_actions, self._observation_shape,
                               gamma=0.99, seed=0)
    # An all-zeros state will produce all-zeros Q values, which will result in
    # action 0 selected by the argmax.
    action = agent.begin_episode(jnp.zeros_like(self._example_state))
    self.assertEqual(0, action)
    # Because we are using a fixed seed we can deterministically guarantee that
    # a state of all ones will pick action 2.
    action = agent.begin_episode(jnp.ones_like(self._example_state))
    self.assertEqual(2, action)

  def test_step_and_end_episode(self):
    agent = mlp_agent.MLPAgent(self._num_actions, self._observation_shape,
                               gamma=0.99, seed=0)
    # Calling step before begin_episode raises an error.
    with self.assertRaises(AttributeError):
      _ = agent.step(0., jnp.zeros_like(self._example_state))
    # Call begin_episode to avoid errors.
    _ = agent.begin_episode(jnp.zeros_like(self._example_state))
    # An all-zeros state will produce all-zeros Q values, which will result in
    # action 0 selected by the argmax.
    action = agent.step(0., jnp.zeros_like(self._example_state))
    self.assertEqual(0, action)
    # Because we are using a fixed seed we can deterministically guarantee that
    # a state of all ones will pick action 2.
    action = agent.step(0., jnp.ones_like(self._example_state))
    self.assertEqual(2, action)
    # end_episode doesn't do anything (it exists to conform to the Agent
    # interface). This next line just checks that it runs without problems.
    agent.end_episode(0., True)

  def test_agent_does_not_train_in_eval_mode(self):
    agent = mlp_agent.MLPAgent(self._num_actions, self._observation_shape,
                               gamma=0.99, seed=0)
    agent.set_mode(base_agent.AgentMode.EVAL)

    params_before = agent.optimizer.target
    agent.begin_episode(jnp.ones(self._observation_shape, dtype=jnp.float32))
    agent.step(1.0, jnp.ones(self._observation_shape, dtype=jnp.float32))
    params_after = agent.optimizer.target

    self.assertEqual(params_before, params_after)


if __name__ == '__main__':
  absltest.main()
