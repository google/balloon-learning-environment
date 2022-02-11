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

from typing import Sequence, Union

from balloon_learning_environment.agents import agent
import numpy as np


class AcmeEvalAgent(agent.Agent):
  """Evaluation agent for ACME."""

  def __init__(self, num_actions: int, observation_shape: Sequence[int]):
    self._num_actions = num_actions
    self._observation_shape = observation_shape
    self.set_mode(agent.AgentMode.EVAL)

  def begin_episode(self, observation: np.ndarray) -> int:
    return 0

  def step(self, reward: float, observation: np.ndarray) -> int:
    return 0

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    pass

  def set_mode(self, mode: Union[agent.AgentMode, str]) -> None:
    mode = agent.AgentMode(mode)
    if mode != agent.AgentMode.EVAL:
      raise ValueError('AcmeEvalAgent only supports EVAL mode.')

  def load_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> None:
    pass
