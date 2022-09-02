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

r"""Acme utils.
"""

import functools
from typing import Any, Dict, Optional, Tuple, Type

from acme import adders
from acme import core
from acme import specs
from acme import wrappers
from acme.agents.jax import dqn
from acme.agents.jax import r2d2
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.networks import base
from acme.jax.networks import duelling
from acme.jax.networks import embedding
from acme.wrappers import observation_action_reward
from balloon_learning_environment.agents import marco_polo_exploration
from balloon_learning_environment.agents import networks
from balloon_learning_environment.agents import random_walk_agent
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.utils import units
import dm_env
from flax import linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax



def _balloon_is_within_radius(state: simulator_data.SimulatorState,
                              radius: units.Distance,
                              max_episode_length: int) -> float:
  balloon_state = state.balloon_state
  return (units.relative_distance(balloon_state.x, balloon_state.y) <=
          radius) / max_episode_length


def create_env(is_eval: bool, max_episode_length: int,
               oar_wrapper: bool = False) -> dm_env.Environment:
  """Creates a BLE environment."""
  env = balloon_env.BalloonEnv()
  if is_eval:
    env = balloon_env.BalloonEnv(
        reward_function=functools.partial(
            _balloon_is_within_radius,
            radius=env.radius,
            max_episode_length=max_episode_length))
  env = wrappers.gym_wrapper.GymWrapper(env)
  env = wrappers.step_limit.StepLimitWrapper(
      env, step_limit=max_episode_length)
  env = wrappers.SinglePrecisionWrapper(env)
  if oar_wrapper:
    env = wrappers.ObservationActionRewardWrapper(env)
  return env


class QuantileNetwork(nn.Module):
  """Network used to compute the agent's return quantiles."""
  num_actions: int
  num_layers: int
  hidden_units: int
  num_atoms: int = 51
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    ble_quantile_network = networks.QuantileNetwork(self.num_actions,
                                                    self.num_layers,
                                                    self.hidden_units,
                                                    self.num_atoms,
                                                    self.inputs_preprocessed)
    def batched_network(x):
      return ble_quantile_network(x)
    # Make network batched, since this is what Acme expects.
    output = jax.vmap(batched_network)(x)
    return {'q_dist': output.logits, 'q_values': output.q_values}


class BLETorso(hk.Module):
  """Simple MLP stack used for BLE."""

  def __init__(self):
    super().__init__(name='ble_torso')
    layers = []
    for _ in range(7):
      layers.append(hk.Linear(600))
      layers.append(jax.nn.relu)
    layers.append(hk.Linear(600))
    self._network = hk.Sequential(layers)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 2
    if inputs_rank < 1 or inputs_rank > 2:
      raise ValueError('Expected input BF or F. Got rank %d' % inputs_rank)

    outputs = self._network(inputs)
    if not batched_inputs:
      return jnp.reshape(outputs, [-1])  # [D]
    return outputs  # [B, D]


class R2D2Network(hk.RNNCore):
  """A duelling recurrent network for use with the BLE."""

  def __init__(self, num_actions: int):
    super().__init__(name='r2d2_ble_network')
    self._embed = embedding.OAREmbedding(BLETorso(), num_actions)
    self._core = hk.LSTM(512)
    self._duelling_head = duelling.DuellingMLP(num_actions, hidden_sizes=[512])
    self._num_actions = num_actions

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._embed(inputs)  # [B, D+A+1]
    core_outputs, new_state = self._core(embeddings, state)
    q_values = self._duelling_head(core_outputs)
    return q_values, new_state

  def initial_state(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    embeddings = hk.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    q_values = hk.BatchApply(self._duelling_head)(core_outputs)  # [T, B, A]
    return q_values, new_states


class CombinedActor(core.Actor):
  """Combines Acme's actor with MarcoPoloExploration exploration actor."""

  def __init__(
      self,
      actor: core.Actor,
      exploration_actor: marco_polo_exploration.MarcoPoloExploration,
  ):
    self._actor = actor
    self._exploration_actor = exploration_actor

  def select_action(self, observation: networks_lib.Observation):
    action = self._actor.select_action(observation)
    action = self._exploration_actor.step(0, observation, action)
    return np.array(action, dtype=np.int32)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)
    self._exploration_actor.begin_episode(timestep.observation, 42)

  def observe(self, action: networks_lib.Action,
              next_timestep: dm_env.TimeStep):
    self._actor.observe(action, next_timestep)

  def update(self, wait: bool = False):
    self._actor.update(wait)


def marco_polo_actor(make_actor_fn):
  """Wraps make_actor_fn to include MarcoPoloExploration."""
  def make_actor(
      random_key: networks_lib.PRNGKey,
      policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ):
    original_actor = make_actor_fn(random_key, policy, environment_spec,
                                   variable_source, adder)
    if adder is None:  # eval actor
      return original_actor

    exploration = marco_polo_exploration.MarcoPoloExploration(
        num_actions=3,
        observation_shape=(1099,),
        exploratory_episode_probability=0.8,
        exploratory_agent_constructor=random_walk_agent.RandomWalkAgent)

    return CombinedActor(original_actor, exploration)

  return make_actor


def dqn_logger():
  return None


def create_dqn(params: Dict[str, Any],
               loss_fn_cls: Type[dqn.QrDqn] = dqn.QrDqn):
  """Creates necessary components to run Acme's DQN."""
  use_marco_polo_exploration = params.pop('marco_polo_exploration', False)
  update_period = 4
  target_update_period = 100
  adaptive_learning_rate = params.pop('adaptive_learning_rate', False)

  config = dqn.DQNConfig(**params)
  config.discount = 0.993
  config.n_step = 5
  config.min_replay_size = 500
  config.target_update_period = target_update_period // update_period
  config.adam_eps = 0.00002
  config.max_replay_size = 2000000
  config.batch_size = 32
  config.samples_per_insert = config.batch_size / update_period
  config.prefetch_size = 0  # Somehow prefetching makes it much slower.
  if adaptive_learning_rate:
    config.learning_rate = optax.linear_schedule(
        init_value=2e-6, end_value=4e-7,
        transition_steps=5_000_000 // config.batch_size)
  else:
    config.learning_rate = 2e-6

  num_atoms = 51
  def make_networks(env_spec):
    q_network = QuantileNetwork(num_actions=3, num_layers=8,
                                hidden_units=600, num_atoms=num_atoms)

    dummy_obs = utils.tile_nested(utils.zeros_like(env_spec.observations), 1)
    dqn_network = networks_lib.FeedForwardNetwork(
        lambda key: q_network.init(key, dummy_obs), q_network.apply)
    return dqn_network

  loss_fn = loss_fn_cls(num_atoms=num_atoms, huber_param=1)
  rl_agent = dqn.DQNBuilder(config=config, loss_fn=loss_fn)

  def behavior_policy(dqn_network):
    def policy(params: networks_lib.Params, key: jnp.ndarray,
               observation: jnp.ndarray, epsilon: float) -> jnp.ndarray:
      observation = jnp.expand_dims(observation, axis=0)  # add batch dim
      action_values = dqn_network.apply(params, observation)['q_values']
      action_values = jnp.squeeze(action_values, axis=0)  # remove batch dim
      result = rlax.epsilon_greedy(epsilon).sample(key, action_values)
      return result
    return policy

  def eval_policy(dqn_network):
    def policy(params: networks_lib.Params, key: jnp.ndarray,
               observation: jnp.ndarray, _) -> jnp.ndarray:
      observation = jnp.expand_dims(observation, axis=0)  # add batch dim
      action_values = dqn_network.apply(params, observation)['q_values']
      action_values = jnp.squeeze(action_values, axis=0)  # remove batch dim
      result = rlax.epsilon_greedy(0).sample(key, action_values)
      return result
    return policy

  if use_marco_polo_exploration:
    rl_agent.make_actor = marco_polo_actor(rl_agent.make_actor)
  return rl_agent, config, make_networks, behavior_policy, eval_policy


def make_r2d2_net2work(env_spec: specs.EnvironmentSpec) -> r2d2.R2D2Networks:
  """Builds R2D2 network for the BLE."""
  def make_core_module() -> hk.RNNCore:
    return R2D2Network(env_spec.actions.num_values)

  return networks_lib.make_unrollable_network(env_spec, make_core_module)
