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

"""Evaluation agent for ACME."""

import os.path as osp
from typing import Sequence, Union

from absl import logging
import acme
from acme import specs
from acme.tf import savers
from balloon_learning_environment import acme_utils
from balloon_learning_environment.agents import agent
import chex
import jax
from jax import numpy as jnp
import numpy as np


@chex.dataclass(frozen=True, mappable_dataclass=False)
class _SimpleActorState:
  rng: jnp.ndarray
  epsilon: float


class AcmeEvalAgent(agent.Agent):
  """Evaluation agent for ACME."""

  def __init__(self, num_actions: int, observation_shape: Sequence[int]):
    self._num_actions = num_actions
    self._observation_shape = observation_shape
    self.set_mode(agent.AgentMode.EVAL)

    self._rl_agent, _, dqn_network_fn, _, self._eval_policy = (
        acme_utils.create_dqn({}))

    observation_spec = specs.Array(
        shape=observation_shape,
        dtype=np.float32,
        name='observation')
    action_spec = specs.DiscreteArray(num_values=num_actions,
                                      dtype=np.int32,
                                      name='action')
    reward_spec = specs.Array(shape=(), dtype=float, name='reward')
    discount_spec = specs.BoundedArray(
        shape=(), dtype=float, minimum=0.0, maximum=1.0, name='discount')
    env_spec = acme.specs.EnvironmentSpec(
        observation_spec,
        action_spec,
        reward_spec,
        discount_spec)

    self._dqn_network = dqn_network_fn(env_spec)
    self._learner = self._rl_agent.make_learner(
        jax.random.PRNGKey(0), self._dqn_network, iter([]))
    self._actor = self._rl_agent.make_actor(
        jax.random.PRNGKey(0),
        self._eval_policy(self._dqn_network),
        variable_source=self._learner)
    self._add_actor_state()

  def _add_actor_state(self):
    # Unused, but required by acme.agents.jax.dqn.actor.
    # pylint: disable=protected-access
    self._actor._state = _SimpleActorState(
        rng=jax.random.PRNGKey(0), epsilon=0.0)
    # pylint: enable=protected-access

  def begin_episode(self, observation: np.ndarray) -> int:
    return self._actor.select_action(observation)

  def step(self, reward: float, observation: np.ndarray) -> int:
    return self._actor.select_action(observation)

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    pass

  def set_mode(self, mode: Union[agent.AgentMode, str]) -> None:
    mode = agent.AgentMode(mode)
    if mode != agent.AgentMode.EVAL:
      raise ValueError('AcmeEvalAgent only supports EVAL mode.')

  def load_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    learner = self._rl_agent.make_learner(
        jax.random.PRNGKey(0), self._dqn_network, iter([]))
    checkpointer = savers.Checkpointer({'learner': learner},
                                       time_delta_minutes=30,
                                       subdirectory='learner',
                                       directory=checkpoint_dir,
                                       max_to_keep=400,  # Large enough number.
                                       add_uid=False)
    # pylint: disable=protected-access
    checkpoint_dir = osp.dirname(
        checkpointer._checkpoint_manager.latest_checkpoint)
    # pylint: enable=protected-access
    checkpoint_to_reload = osp.join(checkpoint_dir, f'ckpt-{iteration_number}')
    # Checkpointer always restores the latest agent, so we re-restore here to
    # force a particular iteration number.
    logging.info('Attempting to restore checkpoint: %d',
                 iteration_number)
    # pylint: disable=protected-access
    checkpointer._checkpoint.restore(checkpoint_to_reload)
    # pylint: enable=protected-access
    self._actor = self._rl_agent.make_actor(
        jax.random.PRNGKey(0), self._eval_policy(self._dqn_network),
        variable_source=learner)
    self._add_actor_state()
