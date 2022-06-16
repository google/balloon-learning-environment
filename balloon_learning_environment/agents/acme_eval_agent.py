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
from typing import Sequence, Union, Dict, Any, Optional

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


class AcmeCheckpointer(savers.Checkpointer):
  """Class for loading checkpoints."""

  def restore(self):
    """Overrides this to avoid loading the latest checkpoint by default."""
    # `savers.Checkpointer` always restores the latest agent, so we `pass` it.
    pass

  def restore_checkpoint(self, checkpoint_to_reload: str):
    self._checkpoint.restore(checkpoint_to_reload).expect_partial()

  @property
  def checkpoint_dir(self):
    return osp.dirname(self._checkpoint_manager.latest_checkpoint)


def learner_logger():
  return None


class AcmeEvalAgent(agent.Agent):
  """Evaluation agent for ACME."""

  def __init__(self,
               num_actions: int,
               observation_shape: Sequence[int],
               params: Optional[Dict[str, Any]] = None):
    self._num_actions = num_actions
    self._observation_shape = observation_shape
    self.set_mode(agent.AgentMode.EVAL)
    self._create_agent(params)
    self._setup_actor_learner()

  def _create_agent(self, params: Dict[str, Any]):
    del params
    self._rl_agent, _, self._network_fn, _, self._eval_policy = (
        acme_utils.create_dqn({}))

  def _setup_actor_learner(self):
    observation_spec = specs.Array(
        shape=self._observation_shape,
        dtype=np.float32,
        name='observation')
    action_spec = specs.DiscreteArray(num_values=self._num_actions,
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

    self._dqn_network = self._network_fn(env_spec)
    self._learner = self._rl_agent.make_learner(
        jax.random.PRNGKey(0),
        self._dqn_network,
        iter([]),
        logger_fn=lambda label, steps_key=None, task_instance=None:
        learner_logger(),
        environment_spec=env_spec)
    self._actor = self._rl_agent.make_actor(
        jax.random.PRNGKey(0),
        self._eval_policy(self._dqn_network),
        environment_spec=env_spec,
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
    # Deprecation warning! Future actors and learners may require an env_spec.
    env_spec = None
    learner = self._rl_agent.make_learner(
        jax.random.PRNGKey(0),
        self._dqn_network,
        iter([]),
        logger_fn=lambda label, steps_key=None, task_instance=None:  # pylint:disable=g-long-lambda
        learner_logger(),
        environment_spec=env_spec)
    checkpointer = AcmeCheckpointer(
        {'learner': learner},
        time_delta_minutes=30,
        subdirectory='learner',
        directory=checkpoint_dir,
        max_to_keep=400,  # Large enough number.
        add_uid=False)
    checkpoint_dir = checkpointer.checkpoint_dir
    checkpoint_to_reload = osp.join(checkpoint_dir, f'ckpt-{iteration_number}')
    logging.info('Attempting to restore checkpoint: %d',
                 iteration_number)
    checkpointer.restore_checkpoint(checkpoint_to_reload)
    self._actor = self._rl_agent.make_actor(
        jax.random.PRNGKey(0), self._eval_policy(self._dqn_network), env_spec,
        variable_source=learner)
    self._add_actor_state()
