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

"""A wrapper for training the Dopamine QR-DQN agent."""

import functools
from typing import Callable, Optional, Sequence, Union

from absl import logging
from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents import dopamine_utils
from balloon_learning_environment.agents import exploration
from balloon_learning_environment.agents import marco_polo_exploration  # pylint: disable=unused-import
from dopamine.jax.agents.quantile import quantile_agent
from flax import linen as nn
import gin
import jax.numpy as jnp
import numpy as np


@gin.configurable
class QuantileAgent(agent.Agent, quantile_agent.JaxQuantileAgent):
  """A wrapper for training the Dopamine QR-DQN agent."""

  def __init__(
      self, num_actions: int, observation_shape: Sequence[int],
      network: nn.Module = gin.REQUIRED,
      exploration_wrapper_constructor: Callable[
          [int, Sequence[int]], exploration.Exploration] = gin.REQUIRED,
      seed: Optional[int] = None):
    """Create the Agent.

    This agent enables one to wrap action selection with another agent, such
    as for exploratory policies. The exploratory agent is in charge of deciding
    whether it will pick the action, or the calling QuantileAgent will.

    Args:
      num_actions: Number of actions.
      observation_shape: Shape of input observations.
      network: Network to use for training and inference.
      exploration_wrapper_constructor: Exploration wrapper for action selection.
      seed: Optional seed for the PRNG.
    """
    # Although Python MRO goes from left to right, we call each __init__
    # function explicitly as opposed to using `super()` (which would just call
    # agent.Agent's init) to avoid confusion.
    agent.Agent.__init__(self, num_actions, observation_shape)
    quantile_agent.JaxQuantileAgent.__init__(
        self,
        num_actions,
        observation_shape=observation_shape,
        observation_dtype=jnp.float32,
        stack_size=1,
        network=network,
        seed=seed)
    self._exploration_wrapper = exploration_wrapper_constructor(
        num_actions, observation_shape)

  def begin_episode(self, observation: np.ndarray) -> int:
    # Note(psc): We need to set `self.action` explicitly here (whether the
    # action is coming from the underlying JaxQuantileAgent or from the
    # _exploration_wrapper) so that the action added to the replay buffer
    # corresponds to the action sent back to the environment. This is
    # because in JaxQuantileAgent, the action is only added to the replay
    # buffer at the *next* step (as it needs to wait for the reward), and
    # at that point will use the action stored in `self.action`.
    self.action = quantile_agent.JaxQuantileAgent.begin_episode(
        self, observation)
    if self.eval_mode:
      return self.action

    self.action = self._exploration_wrapper.begin_episode(
        observation, self.action)
    return self.action

  def step(self, reward: float, observation: np.ndarray) -> int:
    # Note(psc): See note in `begin_episode` on the importance of setting
    # `self.action` to the action sent back to the environment.
    self.action = quantile_agent.JaxQuantileAgent.step(
        self, reward, observation)
    if self.eval_mode:
      return self.action

    self.action = self._exploration_wrapper.step(
        reward, observation, self.action)
    return self.action

  def _train_step(self):
    # We override this method to log using flax's (eager) tensorboard.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        (self.optimizer_state, self.online_params,
         loss, mean_loss) = quantile_agent.train(
             self.network_def,
             self.online_params,
             self.target_network_params,
             self.optimizer,
             self.optimizer_state,
             self.replay_elements['state'],
             self.replay_elements['action'],
             self.replay_elements['next_state'],
             self.replay_elements['reward'],
             self.replay_elements['terminal'],
             self._kappa,
             self._num_atoms,
             self.cumulative_gamma)
        if self._replay_scheme == 'prioritized':
          probs = self.replay_elements['sampling_probabilities']
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))
          loss = loss_weights * loss
          mean_loss = jnp.mean(loss)

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          self.summary_writer.scalar('QuantileLoss', mean_loss,
                                     self.training_steps)
          self.summary_writer.flush()

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    quantile_agent.JaxQuantileAgent.end_episode(self, reward, terminal)

  def set_mode(self, mode: Union[agent.AgentMode, str]) -> None:
    mode = agent.AgentMode(mode)
    if mode == agent.AgentMode.TRAIN:
      self.eval_mode = False
    else:
      self.eval_mode = True

  def save_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    """Checkpoint agent parameters as a pickled dict."""
    dopamine_utils.save_checkpoint(
        checkpoint_dir, iteration_number,
        functools.partial(quantile_agent.JaxQuantileAgent.bundle_and_checkpoint,
                          self))

  def load_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    """Checkpoint agent parameters as a pickled dict."""
    dopamine_utils.load_checkpoint(
        checkpoint_dir, iteration_number,
        functools.partial(quantile_agent.JaxQuantileAgent.unbundle, self))

  def reload_latest_checkpoint(self, checkpoint_dir: str) -> int:
    latest_episode = dopamine_utils.get_latest_checkpoint(checkpoint_dir)
    if latest_episode < 0:
      logging.warning('Unable to reload checkpoint at %s', checkpoint_dir)
      return -1
    try:
      self.load_checkpoint(checkpoint_dir, latest_episode)
      logging.info('Will restart training from episode %d', latest_episode)
      return latest_episode
    except ValueError:
      logging.warning('Unable to reload checkpoint at %s', checkpoint_dir)
      return -1
