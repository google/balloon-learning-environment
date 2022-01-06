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

r"""A simple multi-layer perceptron (MLP) agent.

This agent learns a mapping from states to Q-values using simple SARSA updates.
The approximant to Q is a multi-layer perceptron, where the number of layers
and hidden units is configurable with `gin_bindings`.

For example, to run this agent with 3 layers of 64 units, pass the following
flags to the main runtime binary:
  ```
  --gin_bindings="MLPNetwork.num_layers=3" \
  --gin_bindings="MLPNetwork.hidden_units=64"
  ```
"""

import functools
import time
from typing import Any, Sequence, Union

from absl import logging
from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents import networks
from flax import optim
import gin
import jax
import jax.numpy as jnp
import numpy as np


@gin.configurable
def create_optimizer(learning_rate: float = 0.001):
  """Create an SGD optimizer for training."""
  return optim.GradientDescent(learning_rate=learning_rate)


@functools.partial(jax.jit, static_argnums=0)
def select_action(network_def: Any,
                  network_params: np.ndarray,
                  state: Any) -> int:
  """Select an action greedily from network."""
  return jnp.argmax(network_def.apply(network_params, state))


@functools.partial(jax.jit, static_argnums=(0, 7))
def train(network_def: Any,
          optimizer: Any,
          state: Any,
          action: int,
          reward: float,
          next_state: Any,
          next_action: int,
          gamma: float):
  """Run a single SARSA-update."""
  def loss_fn(params):
    q_val = network_def.apply(params, state)[action]
    next_action_val = network_def.apply(params, next_state)[next_action]
    target = reward + gamma * next_action_val
    return (q_val - target)**2
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return loss, optimizer


@gin.configurable
class MLPAgent(agent.Agent):
  """An agent using a simple MLP network."""

  def __init__(self, num_actions: int, observation_shape: Sequence[int],
               gamma: float = 0.9,
               seed: Union[int, None] = None):
    super().__init__(num_actions, observation_shape)
    self._gamma = gamma
    seed = int(time.time() * 1e6) if seed is None else seed
    rng = jax.random.PRNGKey(seed)
    self.network_def = networks.MLPNetwork(num_actions=(self._num_actions))
    example_state = jnp.zeros(observation_shape)
    network_params = self.network_def.init(rng, example_state)
    optimizer_def = create_optimizer()
    self.optimizer = optimizer_def.create(network_params)

    self._mode = agent.AgentMode('train')

  @property
  def network(self):
    return self.optimizer.target

  def begin_episode(self, observation: np.ndarray) -> int:
    action = select_action(self.network_def,
                           self.network,
                           observation)
    self.last_state = observation
    self.last_action = action
    return action

  def step(self, reward: float, observation: np.ndarray) -> int:
    action = select_action(self.network_def,
                           self.network,
                           observation)

    if self._mode == agent.AgentMode.TRAIN:
      loss, self.optimizer = train(self.network_def,
                                   self.optimizer,
                                   self.last_state,
                                   self.last_action,
                                   reward,
                                   observation,
                                   action,
                                   self._gamma)
      logging.info('Loss: %f', loss)

    self.last_state = observation
    self.last_action = action
    return action

  def end_episode(self, reward: float, terminal: bool) -> None:
    pass

  def set_mode(self, mode: Union[agent.AgentMode, str]) -> None:
    self._mode = agent.AgentMode(mode)
