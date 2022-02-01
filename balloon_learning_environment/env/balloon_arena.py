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

"""A Balloon Arena.

A balloon arena contains the logic for flying a balloon in a wind field.
"""

import abc
import datetime as dt
import math
import time
from typing import Callable, Optional, Union

from balloon_learning_environment.env import features
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import stable_init
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import sampling
from balloon_learning_environment.utils import units
import jax
import jax.numpy as jnp
import numpy as np


class BalloonArenaInterface(abc.ABC):
  """An interface for a balloon arena.

  The balloon arena is the "simulator" for flying stratospheric balloons.
  As such, and child class should encapsulate all functionality and data
  involved in flying balloons, but not the reinforcement learning problem
  (which is encapsulated by the BalloonEnv).
  """

  @abc.abstractmethod
  def reset(self, seed: Optional[int] = None) -> np.ndarray:
    """Resets the arena.

    Args:
      seed: An optional seed for resetting the arena.

    Returns:
      The first observation from the newly reset simulator as a numpy array.
    """

  @abc.abstractmethod
  def step(self, action: control.AltitudeControlCommand) -> np.ndarray:
    """Steps the simulator.

    Args:
      action: The balloon control to apply.

    Returns:
      The observation from the simulator as a numpy array.
    """

  @abc.abstractmethod
  def get_simulator_state(self) -> simulator_data.SimulatorState:
    """Gets the current simulator state.

    This should return the full simulator state so that it can be used for
    checkpointing.

    Returns:
      The simulator state.
    """

  @abc.abstractmethod
  def set_simulator_state(self,
                          new_state: simulator_data.SimulatorState) -> None:
    """Sets the simulator state.

    This should fully restore the simulator state so that it can restore
    from a checkpoint.

    Args:
      new_state: The state to set the simulator to.
    """

  @abc.abstractmethod
  def get_balloon_state(self) -> balloon.BalloonState:
    """Gets the balloon state.

    Returns:
      The current balloon state.
    """

  @abc.abstractmethod
  def set_balloon_state(self, new_state: balloon.BalloonState) -> None:
    """Sets the baloon state.

    Args:
      new_state: The state to set the baloon to.
    """

  @abc.abstractmethod
  def get_measurements(self) -> simulator_data.SimulatorObservation:
    """Gets measurements from the arena.

    This is what a controller may feasibly use to control a balloon.

    Returns:
      Noisy sensor readings of the current state.
    """


class BalloonArena(BalloonArenaInterface):
  """A BalloonArena in which a balloon flies in some wind field."""

  def __init__(self,
               feature_constructor_factory: Callable[
                   [wind_field.WindField, standard_atmosphere.Atmosphere],
                   features.FeatureConstructor],
               wind_field_instance: wind_field.WindField = wind_field
               .SimpleStaticWindField(),
               seed: Optional[int] = None):
    """BalloonArena constructor.

    Args:
      feature_constructor_factory: A factory that when called returns an
        object that constructs feature vectors from observations. The factory
        takes a wind field (WindField) and an initial observation from the
        simulator (SimulatorObservation).
      wind_field_instance: A WindField to use in the simulation.
      seed: An optional seed for the arena. If it is not specified, it will be
        seeded based on the system time.
    """
    self._feature_constructor_factory = feature_constructor_factory
    self._wind_field = wind_field_instance
    self._step_duration = constants.AGENT_TIME_STEP
    self._rng = None  # Set in reset method
    self._balloon: balloon.Balloon  # Initialized in reset.

    # Atmosphere will be seeded with a different key in reset.
    self._atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))

    # TODO(joshgreaves): Set these with gin config
    self._alpha = 1.2
    self._beta = 2.0

    # We call reset here to ensure the arena can always be run without
    # an explicit call to reset. However, the preferred way to run the
    # arena is to call reset immediately to return the intiial observation.
    self.reset(seed)

  def reset(self, seed: Union[int, jnp.ndarray, None] = None) -> np.ndarray:
    if isinstance(seed, int):
      self._rng = jax.random.PRNGKey(seed)
    elif isinstance(seed, (np.ndarray, jnp.ndarray)):
      self._rng = seed
    else:
      # Seed with time in microseconds
      self._rng = jax.random.PRNGKey(int(time.time() * 1e6))

    self._rng, atmosphere_key, time_key = jax.random.split(self._rng, 3)
    self._atmosphere.reset(atmosphere_key)
    start_date_time = sampling.sample_time(time_key)
    self._balloon = self._initialize_balloon(start_date_time)
    assert self._balloon.state.status == balloon.BalloonStatus.OK

    self._rng, wind_field_key = jax.random.split(self._rng, 2)
    self._wind_field.reset(wind_field_key, start_date_time)

    self.feature_constructor = self._feature_constructor_factory(
        self._wind_field, self._atmosphere)
    self.feature_constructor.observe(self.get_measurements())
    return self.feature_constructor.get_features()

  def step(self, action: control.AltitudeControlCommand) -> np.ndarray:
    """Simulates the effects of choosing the given action in the system.

    Args:
      action: The action to take in the simulator.

    Returns:
      A feature vector (numpy array) constructed by the feature constructor.
    """
    # Determine the wind at the balloon's location.
    wind_vector = self._get_wind_ground_truth_at_balloon()

    # Simulate the balloon dynamics in the wind field seconds.
    self._balloon.simulate_step(wind_vector, self._atmosphere, action,
                                self._step_duration)

    # At the end of the cycle, make a measurement, and construct features.
    self.feature_constructor.observe(self.get_measurements())
    return self.feature_constructor.get_features()

  def get_simulator_state(self) -> simulator_data.SimulatorState:
    return simulator_data.SimulatorState(self.get_balloon_state(),
                                         self._wind_field,
                                         self._atmosphere)

  def set_simulator_state(self,
                          new_state: simulator_data.SimulatorState) -> None:
    # TODO(joshgreaves): Restore the state of the feature constructor.
    self.set_balloon_state(new_state.balloon_state)
    self._wind_field = new_state.wind_field
    self._atmosphere = new_state.atmosphere

  def get_balloon_state(self) -> balloon.BalloonState:
    return self._balloon.state

  def set_balloon_state(self, new_state: balloon.BalloonState) -> None:
    self._balloon.state = new_state

  def get_measurements(self) -> simulator_data.SimulatorObservation:
    # TODO(joshgreaves): Add noise to observations
    return simulator_data.SimulatorObservation(
        balloon_observation=self.get_balloon_state(),
        wind_at_balloon=self._get_wind_ground_truth_at_balloon())

  def _initialize_balloon(self,
                          start_date_time: dt.datetime) -> balloon.Balloon:
    """Initializes a balloon.

    Initializes a balloon within 200km of the target. The balloon's distance
    from the target is sampled from a beta distribution, while the direction
    (angle) is sampled uniformly. Its pressure is also sampled uniformly
    from all valid pressures.

    Args:
      start_date_time: The starting date and time.
    Returns:
      A new balloon object.
    """
    self._rng, *keys = jax.random.split(self._rng, num=6)

    # Note: Balloon units are in Pa.
    # Sample the starting distance using a beta distribution, within 200km.
    radius = jax.random.beta(keys[0], self._alpha, self._beta).item()
    radius = units.Distance(km=200.0 * radius)
    theta = jax.random.uniform(keys[1], (), minval=0.0, maxval=2.0 * jnp.pi)

    x = math.cos(theta) * radius
    y = math.sin(theta) * radius
    # TODO(bellemare): Latitude in the tropics, otherwise around the world.
    # Does longitude actually affect anything?
    # TODO(joshgreaves): sample_location only samples between -10 and 10 lat.
    latlng = sampling.sample_location(keys[2])

    pressure = sampling.sample_pressure(keys[3], self._atmosphere)
    upwelling_infrared = sampling.sample_upwelling_infrared(keys[4])
    b = balloon.Balloon(
        balloon.BalloonState(
            center_latlng=latlng,
            x=x,
            y=y,
            pressure=pressure,
            date_time=start_date_time,
            upwelling_infrared=upwelling_infrared))
    stable_init.cold_start_to_stable_params(b.state, self._atmosphere)
    return b

  def _get_wind_ground_truth_at_balloon(self) -> wind_field.WindVector:
    """Returns the wind vector at the balloon's current location."""
    return self._wind_field.get_ground_truth(self._balloon.state.x,
                                             self._balloon.state.y,
                                             self._balloon.state.pressure,
                                             self._balloon.state.time_elapsed)
