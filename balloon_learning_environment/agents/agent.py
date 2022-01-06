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

"""Abstract class for Balloon Learning Environment agents."""


import abc
import enum
from typing import Optional, Sequence, Union

from flax.metrics import tensorboard
import numpy as np


class AgentMode(enum.Enum):
  """An enum for the agent mode."""
  TRAIN = 'train'
  EVAL = 'eval'


class Agent(abc.ABC):
  """Abstract class for defining Balloon Learning Environment agents."""

  def __init__(self, num_actions: int, observation_shape: Sequence[int]):
    """Agent constructor.

    A child class should have the same two arguments in its constructor
    in order to work with `train_lib` and `eval_lib`, which it should pass
    to this constructor.

    Args:
      num_actions: The number of actions available in the environment.
      observation_shape: The shape of the observation vector.
    """
    self._num_actions = num_actions
    self._observation_shape = observation_shape
    self.set_mode(AgentMode.TRAIN)

  def get_name(self) -> str:
    """Gets the name of the agent."""
    return self.__class__.__name__

  @abc.abstractmethod
  def begin_episode(self, observation: np.ndarray) -> int:
    """Begins the episode.

    Must be overridden by child class.

    Args:
      observation: The first observation of an episode returned by the
        environment.

    Returns:
      The action to be applied to the environment.
    """

  @abc.abstractmethod
  def step(self, reward: float, observation: np.ndarray) -> int:
    """Steps the agent.

    Must be overridden by child class.

    Args:
      reward: The last reward returned by the environment.
      observation: The last observation returned by the environment.

    Returns:
      A new action to apply to the environment.
    """

  @abc.abstractmethod
  def end_episode(self, reward: float, terminal: bool = True) -> None:
    """Lets the agent know the episode has ended.

    Must be overriden by child class.

    Args:
      reward: The final reward returned by the environment.
      terminal: Whether the episode ended at a terminal state or not. This
        may be False if the episode ended without reaching a terminal state,
        for example in the case that we are using fixed-length episodes.
    """

  def set_summary_writer(
      self, summary_writer: Optional[tensorboard.SummaryWriter]) -> None:
    """Sets a summary writer for logging to tensorboard."""
    self.summary_writer = summary_writer

  def set_mode(self, mode: Union[AgentMode, str]) -> None:
    """Sets the mode of the agent.

    No-op. It is recommended to override this in the child class.

    If set to train, then the agent may train when being stepped.
    However, if set to eval the agent should be fixed for evaluation.

    Args:
      mode: The mode to set the agent to. Accepts either an enum, or the
        string value of the enum.
    """

  def save_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    """If available, save agent parameters to a checkpoint.

    No-op. It is recommended to override this in the child class.

    Args:
      checkpoint_dir: The directory to write the checkpoint to.
      iteration_number: The current iteration number.
    """

  def load_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    """If available, load agent parameters from a checkpoint.

    No-op. It is recommended to override this in the child class.

    Args:
      checkpoint_dir: The directory to load the checkpoint from.
      iteration_number: The current iteration number.
    """

  def reload_latest_checkpoint(self, checkpoint_dir: str) -> int:
    """If available, load agent parameters from the latest checkpoint.

    No-op. It is recommended to override this in the child class.

    Args:
      checkpoint_dir: Directory in which to look for checkpoints.

    Returns:
      Latest checkpoint number found, or -1 if none found.
    """
    del checkpoint_dir
    return -1


class RandomAgent(Agent):
  """An agent that takes uniform random actions."""

  def _random_action(self) -> int:
    return np.random.randint(self._num_actions)

  def begin_episode(self, unused_obs: np.ndarray) -> int:
    return self._random_action()

  def step(self, reward: float, observation: np.ndarray) -> int:
    return self._random_action()

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    pass
