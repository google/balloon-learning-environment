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

"""A wrapper for training the Dopamine QR-DQN agent."""

import functools
from typing import Callable, Optional, Sequence, Union

from absl import logging
from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents import dopamine_utils
from balloon_learning_environment.agents import exploration
from balloon_learning_environment.agents import marco_polo_exploration  # pylint: disable=unused-import
from balloon_learning_environment.agents import perciatelli44
from dopamine.jax.agents.quantile import quantile_agent
import flax
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as np


@gin.configurable(allowlist=['network',
                             'exploration_wrapper_constructor',
                             'checkpoint_duration',
                             'reload_perciatelli'])
class QuantileAgent(agent.Agent, quantile_agent.JaxQuantileAgent):
  """A wrapper for training the Dopamine QR-DQN agent."""

  def __init__(
      self,
      num_actions: int,
      observation_shape: Sequence[int],
      *,  # Everything after this is a keyword-only argument.
      seed: Optional[int] = None,
      network: nn.Module = gin.REQUIRED,
      exploration_wrapper_constructor: Callable[
          [int, Sequence[int]], exploration.Exploration] = gin.REQUIRED,
      checkpoint_duration: Optional[int] = gin.REQUIRED,
      reload_perciatelli: bool = gin.REQUIRED):
    """Create the Agent.

    This agent enables one to wrap action selection with another agent, such
    as for exploratory policies. The exploratory agent is in charge of deciding
    whether it will pick the action, or the calling QuantileAgent will.

    Args:
      num_actions: Number of actions.
      observation_shape: Shape of input observations.
      seed: Optional seed for the PRNG.
      network: Network to use for training and inference.
      exploration_wrapper_constructor: Exploration wrapper for action selection.
      checkpoint_duration: Optional duration of checkpoints for garbage
        collection.
      reload_perciatelli: Whether to reload the weights from the Perciatelli44
        agent.
    """
    self._checkpoint_duration = checkpoint_duration
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
    if reload_perciatelli:
      self.online_params = self.load_perciatelli_weights()
      self.target_network_params = self.online_params
      logging.info('Successfully loaded Perciatelli44 parameters.')

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
    # Get rid of old checkpoints if necessary.
    if self._checkpoint_duration is not None:
      dopamine_utils.clean_up_old_checkpoints(
          checkpoint_dir, iteration_number,
          checkpoint_duration=self._checkpoint_duration)

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

  @staticmethod
  def load_perciatelli_weights() -> flax.core.FrozenDict:
    """Load the Perciatelli weights and convert to a JAX array."""
    sess = perciatelli44.load_perciatelli_session()
    layer_names = [n.name
                   for n in sess.graph.as_graph_def().node
                   if 'Online' in n.name]

    param_dict = {}
    for name in layer_names:
      if not ('weights' in name or 'biases' in name) or 'read' in name:
        continue

      params = sess.run(sess.graph.get_tensor_by_name(f'{name}:0'))
      param_dict[name] = params
    jax_params = {
        'params': {
            'Dense_0': {
                'kernel': param_dict['Online/fully_connected/weights'],
                'bias': param_dict['Online/fully_connected/biases'],
            },
            'Dense_1': {
                'kernel': param_dict['Online/fully_connected_1/weights'],
                'bias': param_dict['Online/fully_connected_1/biases'],
            },
            'Dense_2': {
                'kernel': param_dict['Online/fully_connected_2/weights'],
                'bias': param_dict['Online/fully_connected_2/biases'],
            },
            'Dense_3': {
                'kernel': param_dict['Online/fully_connected_3/weights'],
                'bias': param_dict['Online/fully_connected_3/biases'],
            },
            'Dense_4': {
                'kernel': param_dict['Online/fully_connected_4/weights'],
                'bias': param_dict['Online/fully_connected_4/biases'],
            },
            'Dense_5': {
                'kernel': param_dict['Online/fully_connected_5/weights'],
                'bias': param_dict['Online/fully_connected_5/biases'],
            },
            'Dense_6': {
                'kernel': param_dict['Online/fully_connected_6/weights'],
                'bias': param_dict['Online/fully_connected_6/biases'],
            },
            'Dense_7': {
                'kernel': param_dict['Online/fully_connected_7/weights'],
                'bias': param_dict['Online/fully_connected_7/biases'],
            },
        }
    }
    jax_params = jax.tree_map(jnp.asarray, jax_params)
    return flax.core.FrozenDict(jax_params)
