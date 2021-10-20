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

"""Exploration strategy used in the Nature paper.

Specifically, it interleaves between two phases:
  - RL phase (where the parent agent's actions are maintained).
  - Exploration phase (where a second agent picks actions).
"""

import datetime as dt
import time
from typing import Callable, Optional, Sequence

from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents import exploration
from balloon_learning_environment.utils import constants
import gin
import jax
import numpy as np


_RL_PHASE_LENGTH = dt.timedelta(hours=4)
_EXPLORATORY_PHASE_LENGTH = dt.timedelta(hours=2)


@gin.configurable
class MarcoPoloExploration(exploration.Exploration):
  """Exploration strategy used in the Nature paper."""

  def __init__(self, num_actions: int, observation_shape: Sequence[int],
               exploratory_episode_probability: float = gin.REQUIRED,
               exploratory_agent_constructor: Callable[
                   [int, Sequence[int]], agent.Agent] = gin.REQUIRED,
               seed: Optional[int] = None):
    self._exploratory_agent = exploratory_agent_constructor(
        num_actions, observation_shape)
    self._exploratory_episode_probability = exploratory_episode_probability
    self._exploratory_episode = False
    self._exploratory_phase = False
    self._phase_time_elapsed = dt.timedelta()
    seed = int(time.time() * 1e6) if seed is None else seed
    self._rng = jax.random.PRNGKey(seed)

  def begin_episode(self, observation: np.ndarray, action: int) -> int:
    """Initialize episode, which always starts in RL phase."""
    self._exploratory_agent.begin_episode(observation)
    self._phase_time_elapsed = dt.timedelta()
    rng, self._rng = jax.random.split(self._rng)
    self._exploratory_episode = (
        jax.random.uniform(rng) <= self._exploratory_episode_probability)
    # We always start in the RL phase.
    self._exploratory_phase = False
    return action

  def _phase_expired(self) -> bool:
    if (self._exploratory_phase and
        self._phase_time_elapsed >= _EXPLORATORY_PHASE_LENGTH):
      return True

    if not self._exploratory_phase and self._phase_time_elapsed >= _RL_PHASE_LENGTH:
      return True

    return False

  def _update_phase(self) -> None:
    self._phase_time_elapsed += constants.AGENT_TIME_STEP
    if not self._exploratory_episode:
      return

    if self._phase_expired():
      self._exploratory_phase = not self._exploratory_phase
      self._phase_time_elapsed = dt.timedelta()

  def step(self, reward: float, observation: np.ndarray, action: int) -> int:
    """Return `action` if in RL phase, otherwise query _exploratory_agent."""
    self._update_phase()
    if self._exploratory_phase:
      return self._exploratory_agent.step(reward, observation)

    return action
