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

"""Balloon Learning Environment.

This provides the RL layer on top of the simulator.
"""

import math
import time
from typing import Any, Callable, Mapping, Optional, Text, Tuple, Union, Dict

from balloon_learning_environment.env import balloon_arena
from balloon_learning_environment.env import features
from balloon_learning_environment.env import generative_wind_field  # pylint: disable=unused-import
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.rendering import renderer as randerer_lib
from balloon_learning_environment.utils import transforms
from balloon_learning_environment.utils import units
import gin
import gym
import jax
import numpy as np


# TODO(joshgreaves): Maybe move into its own file.
@gin.configurable
def perciatelli_reward_function(
    simulator_state: simulator_data.SimulatorState,
    *,
    station_keeping_radius_km: float = 50.0,
    reward_dropoff: float = 0.4,
    reward_halflife: float = 100.0) -> float:
  """The reward function used to train Perciatelli44.

  The reward function for the environment returns 1.0 when the balloon is
  with the station keeping radius, and roughly:

    reward_dropoff * 2^(-distance_from_radius / reward_halflife)

  when outside the station keeping radius. That is, the reward immediately
  outside the station keeping radius is reward_dropoff, and the reward
  decays expontentially as the balloon moves further from the radius.

  Args:
    simulator_state: The current state of the simulator to calculate
      reward for.
    station_keeping_radius_km: The desired station keeping radius in km. When
      the balloon is within this radius, the reward is 1.0.
    reward_dropoff: The reward multiplier for when the balloon is outside of
      station keeping range. See reward definition above.
    reward_halflife: The half life of the reward function. See reward
      definition above.

  Returns:
    A reward for the current simulator_state.
  """
  balloon_state = simulator_state.balloon_state
  x, y = balloon_state.x, balloon_state.y
  radius = units.Distance(km=station_keeping_radius_km)

  # x, y are in meters.
  distance = units.relative_distance(x, y)

  # Base reward - distance to station keeping radius.
  if distance <= radius:
    # Reward is 1.0 within the radius.
    reward = 1.0
  else:
    # Exponential decay outside boundary with drop
    # ln(0.5) is approximately -0.69314718056.
    reward = reward_dropoff * math.exp(
        -0.69314718056 / reward_halflife * (distance - radius).kilometers)

  # Power regularization. Only applied when using more power (going down)
  # and there isn't excess energy available.
  if (balloon_state.last_command == control.AltitudeControlCommand.DOWN and
      not balloon_state.excess_energy):
    max_multiplier = 0.95
    penalty_skew = 0.3
    scale = transforms.linear_rescale_with_saturation(
        balloon_state.acs_power.watts, 100.0, 300.0)
    multiplier = max_multiplier - penalty_skew * scale
    reward *= multiplier

  return reward


@gin.configurable
class BalloonEnv(gym.Env):
  """Balloon Learning Environment."""

  def __init__(
      self,
      *,  # All arguments after this are keyword-only.
      station_keeping_radius_km: float = 50.0,
      arena: Optional[balloon_arena.BalloonArenaInterface] = None,
      reward_function: Callable[
          [simulator_data.SimulatorState], float] = perciatelli_reward_function,
      feature_constructor_factory: Callable[
          [wind_field.WindField, standard_atmosphere.Atmosphere],
          features.FeatureConstructor] = features.PerciatelliFeatureConstructor,
      wind_field_factory: Callable[
          [], wind_field.WindField] = generative_wind_field.GenerativeWindField,
      seed: Optional[int] = None,
      renderer: Optional[randerer_lib.Renderer] = None):
    """Constructs a Balloon Learning Environment Station Keeping Environment.

    Args:
      station_keeping_radius_km: The desired station keeping radius in km.
      arena: A balloon arena (simulator) to wrap. If set to None, it will use
        the default balloon arena.
      reward_function: A function that takes the current simulator state
        and returns a scalar reward.
      feature_constructor_factory: A callable which returns a new
        FeatureConstructor object when called. The factory takes a forecast
        (WindField) and an initial observation from the simulator
        (SimulatorObservation).
      wind_field_factory: A callable which returns a new WindField object.
      seed: A PRNG seed for the environment.
      renderer: An optional renderer for rendering flight paths/simulator state.
    """
    self.radius = units.Distance(km=station_keeping_radius_km)
    self._reward_fn = reward_function
    self._feature_constructor_factory = feature_constructor_factory
    self._global_iteration = 0

    if arena is None:
      self.arena = balloon_arena.BalloonArena(self._feature_constructor_factory,
                                              wind_field_factory())
    else:
      self.arena = arena

    self._renderer = renderer
    if renderer is not None:
      self.metadata = {'render.modes': self._renderer.render_modes}

    # Use time in microseconds if a seed is not supplied.
    self.reset(seed=seed if seed is not None else int(time.time() * 1e6))

  def step(self,
           action: int) -> Tuple[np.ndarray, float, bool, Mapping[str, Any]]:
    """Applies an action and steps the environment.

    Args:
      action: An integer action corresponding to AlititudeControlCommand in
        env/balloon.py.

    Returns:
      An (observation, reward, terminal, info) tuple.
    """
    command = control.AltitudeControlCommand(action)
    observation = self.arena.step(command)
    assert isinstance(observation, np.ndarray)

    simulator_state = self.arena.get_simulator_state()

    if self._renderer is not None:
      self._renderer.step(simulator_state)

    # Prepare reward
    reward = self._reward_fn(simulator_state)

    # Prepare is_terminal
    info = self._get_info(simulator_state.balloon_state)
    is_terminal = (
        info['out_of_power']
        or info['envelope_burst']
        or info['zeropressure']
    )

    self._global_iteration += 1

    return observation, reward, is_terminal, info

  def reset(
      self,
      *,
      seed: Optional[int] = None,
      return_info: bool = False
  ) -> Union[np.ndarray, Tuple[np.ndarray, Mapping[str, Any]]]:
    """Resets the environment.

    Args:
      seed: Seed to re-seed environment on reset.
      return_info: return info dictionary with initial observation.

    Returns:
      If `return_info` is True:
        A tuple consisting of initial_observation and the info dictionary.
      Otherwise:
        The initial observation.
    """
    if seed is not None:
      self.seed(seed)

    self._rng, arena_rng = jax.random.split(self._rng)
    observation = self.arena.reset(arena_rng)

    if self._renderer is not None:
      self._renderer.reset()
      self._renderer.step(self.arena.get_simulator_state())

    if return_info:
      simulator_state = self.get_simulator_state()
      info = self._get_info(simulator_state.balloon_state)

      return observation, info
    else:
      return observation

  def render(self, mode: str = 'human') -> Union[None, np.ndarray, Text]:
    """Renders a frame.

    Args:
      mode: One of `human`, `rgb_array`, or `ansi`. `human`
        corresponds to rendering directly to the screen. `rgb_array` renders to
        a numpy array and returns it. `ansi` renders to a string or StringIO
        object.

    Returns:
      None, a numpy array of rgb data, or a Text object, depending on the mode.

    Raises:
      ValueError: Propagated from the internal renderer if mode is not in
        self.metadata['render.modes'] i.e. not implemented in the renderer.
    """
    if self._renderer is None:
      return None

    return self._renderer.render(mode)

  def close(self) -> None:
    # Nothing to cleanup.
    pass

  def seed(self, seed: int) -> None:
    """Seeds the environment."""
    self._rng = jax.random.PRNGKey(seed)

  @property
  def unwrapped(self) -> gym.Env:
    return self

  @property
  def action_space(self) -> gym.spaces.Discrete:
    """Gets the action space."""
    return gym.spaces.Discrete(3)

  @property
  def observation_space(self) -> gym.Space:
    """Gets the observation space."""
    return self.arena.feature_constructor.observation_space

  @property
  def reward_range(self) -> Tuple[float, float]:
    """Gets the reward range."""
    return (0.0, 1.0)

  def get_simulator_state(self) -> simulator_data.SimulatorState:
    """Gets the simulator state."""
    return self.arena.get_simulator_state()

  def _get_info(self, balloon_state: balloon.BalloonState) -> Dict[str, Any]:
    out_of_power = balloon_state.status == balloon.BalloonStatus.OUT_OF_POWER
    envelope_burst = balloon_state.status == balloon.BalloonStatus.BURST
    zeropressure = balloon_state.status == balloon.BalloonStatus.ZEROPRESSURE

    return {
        'out_of_power': out_of_power,
        'envelope_burst': envelope_burst,
        'zeropressure': zeropressure,
        'time_elapsed': balloon_state.time_elapsed
    }

  def __str__(self) -> str:
    return 'BalloonEnv'

  def __enter__(self) -> gym.Env:
    return self

  def __exit__(self, *args: Any) -> bool:
    self.close()
    return False  # Reraise any exceptions


def register_env():
  gym.register(
      id='BalloonLearningEnvironment-v0',
      entry_point='balloon_learning_environment.env.balloon_env:BalloonEnv')


register_env()
