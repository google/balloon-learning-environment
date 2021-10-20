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

"""An exploratory agent that selects actions based on a random walk.

Note that this class assumes the features passed in correspond to the
Perciatelli features (see balloon_learning_environment.env.features).
"""


import datetime as dt
import time
from typing import Optional, Sequence

from balloon_learning_environment.agents import agent
from balloon_learning_environment.env import features
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import sampling
import gin
import jax
import numpy as np


_PERCIATELLI_FEATURES_SHAPE = (1099,)  # Expected shape of Perciatelli features.
_HYSTERESIS = 100  # In Pascals.
_STDDEV = 0.1666  # ~ 10 [Pa/min].


# Although this class does not have any gin-configurable parameters, it is
# decorated as gin-configurable so it can be injected into other classes
# (e.g. MarcoPoloExploration).
@gin.configurable
class RandomWalkAgent(agent.Agent):
  """An exploratory agent that selects actions based on a random walk."""

  def __init__(self, num_actions: int, observation_shape: Sequence[int],
               seed: Optional[int] = None):
    del num_actions
    del observation_shape
    seed = int(time.time() * 1e6) if seed is None else seed
    self._rng = jax.random.PRNGKey(seed)
    self._time_elapsed = dt.timedelta()
    self._sample_new_target_pressure()

  def _sample_new_target_pressure(self):
    self._rng, rng = jax.random.split(self._rng)
    self._target_pressure = sampling.sample_pressure(rng)

  def _select_action(self, features_as_vector: np.ndarray) -> int:
    assert features_as_vector.shape == _PERCIATELLI_FEATURES_SHAPE
    balloon_pressure = features.NamedPerciatelliFeatures(
        features_as_vector).balloon_pressure
    # Note: higher pressures means lower altitude.
    if balloon_pressure - _HYSTERESIS > self._target_pressure:
      return control.AltitudeControlCommand.UP

    if balloon_pressure + _HYSTERESIS < self._target_pressure:
      return control.AltitudeControlCommand.DOWN

    return control.AltitudeControlCommand.STAY

  def begin_episode(self, observation: np.ndarray) -> int:
    self._time_elapsed = dt.timedelta()
    self._sample_new_target_pressure()
    return self._select_action(observation)

  def step(self, reward: float, observation: np.ndarray) -> int:
    del reward
    # Advance time_elapsed.
    self._time_elapsed += constants.AGENT_TIME_STEP
    # Update target pressure. This is essentially a random walk between
    # altitudes by sampling from zero-mean Gaussian noise, where the amount of
    # variance is proportional to the amount of time (in seconds) that has
    # elapsed since the last time it was updated.
    self._rng, rng = jax.random.split(self._rng)
    self._target_pressure += (
        self._time_elapsed.total_seconds() * _STDDEV * jax.random.normal(rng))
    return self._select_action(observation)

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    pass
