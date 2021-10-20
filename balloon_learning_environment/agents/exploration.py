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

"""Base class for exploration modules which can be used by the agents.

An Agent can wrap an Exploration module during its action selection.
So rather than simply issuing
  `return action`
It can issue:
  `return self.exploration_wrapper(action)`
"""

from typing import Sequence
import numpy as np


class Exploration(object):
  """Base class for an exploration module; this wrapper is a no-op."""

  def __init__(self, unused_num_actions: int,
               unused_observation_shape: Sequence[int]):
    pass

  def begin_episode(self, observation: np.ndarray, a: int) -> int:
    """Returns the same action passed by the agent."""
    del observation
    return a

  def step(self, reward: float, observation: np.ndarray, a: int) -> int:
    """Returns the same action passed by the agent."""
    del reward
    del observation
    return a
