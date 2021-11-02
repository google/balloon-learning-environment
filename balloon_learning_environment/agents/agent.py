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

"""Abstract class for Balloon Learning Environment agents."""


import abc
import enum
from typing import Optional, Sequence, Union

from flax.metrics import tensorboard
import numpy as np


class AgentMode(enum.Enum):
  TRAIN = 'train'
  EVAL = 'eval'


class Agent(abc.ABC):
  """Abstract class for defining Balloon Learning Environment agents."""

  def __init__(self, num_actions: int, observation_shape: Sequence[int]):
    self._num_actions = num_actions
    self._observation_shape = observation_shape
    self.set_mode(AgentMode.TRAIN)

  def get_name(self) -> str:
    return self.__class__.__name__

  @abc.abstractmethod
  def begin_episode(self, observation: np.ndarray) -> int:
    pass

  @abc.abstractmethod
  def step(self, reward: float, observation: np.ndarray) -> int:
    pass

  @abc.abstractmethod
  def end_episode(self, reward: float, terminal: bool = True) -> None:
    pass

  def set_summary_writer(
      self, summary_writer: Optional[tensorboard.SummaryWriter]) -> None:
    self.summary_writer = summary_writer

  def set_mode(self, mode: Union[AgentMode, str]) -> None:
    """Sets the mode of the agent.

    If set to train, then the agent may train when being stepped.
    However, if set to eval the agent should be fixed for evaluation.

    Args:
      mode: The mode to set the agent to. Accepts either an enum, or the
        string value of the enum.
    """
    pass

  def save_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    """If available, save agent parameters to a checkpoint."""
    pass

  def load_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    """If available, load agent parameters from a checkpoint."""
    pass

  def reload_latest_checkpoint(self, checkpoint_dir: str) -> int:
    """If available, load agent parameters from the latest checkpoint.

    Args:
      checkpoint_dir: Directory where to look for checkpoints.

    Returns:
      Latest checkpoint number found, or -1 if none found.
    """
    del checkpoint_dir
    return -1


class RandomAgent(Agent):
  """A random agent."""

  def _random_action(self) -> int:
    return np.random.randint(self._num_actions)

  def begin_episode(self, unused_obs: np.ndarray) -> int:
    return self._random_action()

  def step(self, reward: float, observation: np.ndarray) -> int:
    return self._random_action()

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    pass
