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

"""A wrapper for training the Dopamine DQN agent."""

import functools
from typing import Optional, Sequence, Union

from absl import logging
from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents import dopamine_utils
from dopamine.jax.agents.dqn import dqn_agent
from flax import linen as nn
import gin
import jax.numpy as jnp
import numpy as np


@gin.configurable(allowlist=['network', 'checkpoint_duration'])
class DQNAgent(agent.Agent, dqn_agent.JaxDQNAgent):
  """A wrapper for training the Dopamine DQN agent."""

  def __init__(self,
               num_actions: int,
               observation_shape: Sequence[int],
               *,  # Everything after this is a keyword-only argument.
               seed: Optional[int] = None,
               network: nn.Module = gin.REQUIRED,
               checkpoint_duration: Optional[int] = gin.REQUIRED):
    """Create the DQN Agent.

    Args:
      num_actions: Number of actions.
      observation_shape: Shape of input observations.
      seed: Optional seed for the PRNG.
      network: Network to use for training and inference.
      checkpoint_duration: Optional duration of checkpoints for garbage
        collection.
    """
    self._checkpoint_duration = checkpoint_duration
    # Although Python MRO goes from left to right, we call each __init__
    # function explicitly as opposed to using `super()` (which would just call
    # agent.Agent's init) to avoid confusion.
    agent.Agent.__init__(self, num_actions, observation_shape)
    dqn_agent.JaxDQNAgent.__init__(
        self,
        num_actions,
        observation_shape=observation_shape,
        observation_dtype=jnp.float32,
        stack_size=1,
        network=functools.partial(network, is_dopamine=True),
        seed=seed)

  def begin_episode(self, observation: np.ndarray) -> int:
    return dqn_agent.JaxDQNAgent.begin_episode(self, observation)

  def step(self, reward: float, observation: np.ndarray) -> int:
    return dqn_agent.JaxDQNAgent.step(self, reward, observation)

  def _train_step(self):
    # We override this method to log using flax's (eager) tensorboard.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self.optimizer_state, self.online_params, loss = dqn_agent.train(
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
            self.cumulative_gamma,
            self._loss_type)

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          self.summary_writer.scalar('HuberLoss', loss, self.training_steps)
          self.summary_writer.flush()

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    dqn_agent.JaxDQNAgent.end_episode(self, reward, terminal)

  def set_mode(self, mode: Union[agent.AgentMode, str]) -> None:
    mode = agent.AgentMode(mode)
    if mode == agent.AgentMode.TRAIN:
      self.eval_mode = False
    else:
      self.eval_mode = True

  def save_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    """Checkpoint agent parameters as a pickled dict."""
    # Try to create checkpoint directory if it doesn't exist.
    dopamine_utils.save_checkpoint(
        checkpoint_dir, iteration_number,
        functools.partial(dqn_agent.JaxDQNAgent.bundle_and_checkpoint,
                          self))
    # Get rid of old checkpoints if necessary.
    if self._checkpoint_duration is not None:
      dopamine_utils.clean_up_old_checkpoints(
          checkpoint_dir, iteration_number,
          checkpoint_duration=self._checkpoint_duration)

  def load_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    """Checkpoint agent parameters as a pickled dict."""
    dopamine_utils.load_checkpoint(
        checkpoint_dir, iteration_number,
        functools.partial(dqn_agent.JaxDQNAgent.unbundle, self))

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
