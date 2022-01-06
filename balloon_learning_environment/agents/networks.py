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

"""A common set of networks available for agents."""

from absl import logging
from dopamine.discrete_domains import atari_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp


@gin.configurable
class MLPNetwork(nn.Module):
  """A simple MLP network."""
  num_actions: int
  num_layers: int = gin.REQUIRED
  hidden_units: int = gin.REQUIRED
  is_dopamine: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    # This method sets up the MLP for inference, using the specified number of
    # layers and units.
    logging.info('Creating MLP network with %d layers and %d hidden units',
                 self.num_layers, self.hidden_units)

    # Network initializer.
    kernel_initializer = jax.nn.initializers.glorot_uniform()
    x = x.astype(jnp.float32)  # Convert to JAX float32 type.
    x = x.reshape(-1)  # Flatten.

    # Pass through the desired number of hidden layers (we do this for
    # one less than `self.num_layers`, as `self._final_layer` counts as one).
    for _ in range(self.num_layers - 1):
      x = nn.Dense(features=self.hidden_units,
                   kernel_init=kernel_initializer)(x)
      x = nn.relu(x)

    # The final layer will output a value for each action.
    q_values = nn.Dense(features=self.num_actions,
                        kernel_init=kernel_initializer)(x)

    if self.is_dopamine:
      q_values = atari_lib.DQNNetworkType(q_values)
    return q_values


@gin.configurable
class QuantileNetwork(nn.Module):
  """Network used to compute the agent's return quantiles."""
  num_actions: int
  num_layers: int = gin.REQUIRED
  hidden_units: int = gin.REQUIRED
  num_atoms: int = 51  # Normally set by JaxQuantileAgent.
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    # This method sets up the MLP for inference, using the specified number of
    # layers and units.
    logging.info('Creating MLP network with %d layers, %d hidden units, and '
                 '%d atoms', self.num_layers, self.hidden_units, self.num_atoms)

    # Network initializer.
    kernel_initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    x = x.astype(jnp.float32)  # Convert to JAX float32 type.
    x = x.reshape(-1)  # Flatten.

    # Pass through the desired number of hidden layers (we do this for
    # one less than `self.num_layers`, as `self._final_layer` counts as one).
    for _ in range(self.num_layers - 1):
      x = nn.Dense(features=self.hidden_units,
                   kernel_init=kernel_initializer)(x)
      x = nn.relu(x)

    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=kernel_initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.mean(logits, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)
