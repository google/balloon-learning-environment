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

"""Unit test helpers."""

import dataclasses
import datetime as dt
from typing import Callable, Optional, Sequence

from absl.testing import absltest
from balloon_learning_environment.env import balloon_arena
from balloon_learning_environment.env import features
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import stable_init
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.rendering import renderer as randerer_lib
from balloon_learning_environment.utils import units
import gin
import jax
import jax.numpy as jnp
import numpy as np

import s2sphere as s2

START_DATE_TIME = units.datetime(2013, 3, 25, 9, 25, 32)


def bind_environment_gin_parameters(
    *,  # All arguments after this are keyword-only.
    station_keeping_radius_km: float = 50.0,
    reward_dropoff: float = 0.4,
    reward_halflife: float = 100.0,
    arena: Optional[balloon_arena.BalloonArenaInterface] = None,
    feature_constructor_factory: Callable[
        [wind_field.WindField],
        features.FeatureConstructor] = features.PerciatelliFeatureConstructor,
    wind_field_factory: Callable[
        [], wind_field.WindField] = wind_field.SimpleStaticWindField,
    seed: Optional[int] = 1,
    renderer: Optional[randerer_lib.Renderer] = None) -> None:
  """Binds gin parameters for BalloonEnv.

  Args:
    station_keeping_radius_km: The desired station keeping radius in km. When
      the balloon is within this radius, the reward is 1.0.
    reward_dropoff: The reward multiplier for when the balloon is outside of
      station keeping range. See reward definition above.
    reward_halflife: The half life of the reward function. See reward definition
      above.
    arena: A balloon arena (simulator) to wrap. If set to None, it will use the
      default balloon arena.
    feature_constructor_factory: A callable which returns a new
      FeatureConstructor object when called. The factory takes a forecast
      (WindField) and an initial observation from the simulator
      (SimulatorObservation).
    wind_field_factory: A callable which returns a new WindField object.
    seed: A PRNG seed for the environment. Defaults to 1, to encourage
      deterministic tests. If environment randomness is required, set to
      None.
    renderer: An optional renderer.
  """
  gin.bind_parameter('perciatelli_reward_function.station_keeping_radius_km',
                     station_keeping_radius_km)
  gin.bind_parameter('perciatelli_reward_function.reward_dropoff',
                     reward_dropoff)
  gin.bind_parameter('perciatelli_reward_function.reward_halflife',
                     reward_halflife)
  gin.bind_parameter('BalloonEnv.arena', arena)
  gin.bind_parameter('BalloonEnv.feature_constructor_factory',
                     feature_constructor_factory)
  gin.bind_parameter('BalloonEnv.wind_field_factory', wind_field_factory)
  gin.bind_parameter('BalloonEnv.seed', seed)
  gin.bind_parameter('BalloonEnv.renderer', renderer)


def create_balloon(
    x: units.Distance = units.Distance(m=0.0),
    y: units.Distance = units.Distance(m=0.0),
    center_lat: float = 0.0,
    center_lng: float = 0.0,
    pressure: float = 7_000.0,
    power_percent: float = 0.95,
    date_time: Optional[dt.datetime] = None,
    time_elapsed: Optional[dt.timedelta] = None,
    power_safety_layer_enabled: bool = True,
    use_stable_init: bool = True,
    upwelling_infrared: float = 250.0,
    atmosphere: Optional[standard_atmosphere.Atmosphere] = None,
) -> balloon.Balloon:
  """Creates a balloon object for easy testing."""
  date_time = date_time if date_time is not None else START_DATE_TIME
  time_elapsed = time_elapsed if time_elapsed is not None else dt.timedelta()
  b = balloon.Balloon(
      balloon.BalloonState(
          center_latlng=s2.LatLng.from_degrees(center_lat, center_lng),
          x=x,
          y=y,
          pressure=pressure,
          date_time=date_time,
          power_safety_layer_enabled=power_safety_layer_enabled,
          time_elapsed=time_elapsed,
          upwelling_infrared=upwelling_infrared))
  b.state.battery_charge = power_percent * b.state.battery_capacity

  if use_stable_init:
    if atmosphere is None:
      raise ValueError('Must supply an Atmosphere object if using stable init.')
    stable_init.cold_start_to_stable_params(b.state, atmosphere)

  return b


# TODO(joshgreaves): This is quite complex - write a test for it.
def compare_balloon_states(b1: balloon.BalloonState,
                           b2: balloon.BalloonState,
                           check_not_equal: Sequence[str] = ()):
  """Function for comparing balloon states.

  Args:
    b1: First BalloonState.
    b2: Second BalloonState.
    check_not_equal: A sequence of strings. Each string corresponds to a field
      in BaloonStates to check if they are not equal. If not empty or None, not
      equal will be asserted of the values or objects corresponding to these
      fields.
  """
  # pylint: disable=protected-access
  for field in dataclasses.fields(balloon.BalloonState):
    key = field.name
    x = b1.__dict__[key]
    y = b2.__dict__[key]

    if 'safety_layer' in key:
      continue

    if isinstance(x, (np.ndarray, jnp.ndarray)):
      absltest.TestCase().assertIsInstance(y, (np.ndarray, jnp.ndarray))
      if key in check_not_equal:
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, x, y)
      elif not check_not_equal:
        np.testing.assert_equal(x, y)

    else:
      if key in check_not_equal:
        if field != 'power':
          absltest.TestCase().assertNotEqual(x, y)
        else:
          absltest.TestCase().assertNotEqual(
              x.battery_charge, y.battery_charge)
      elif not check_not_equal:
        if key != 'power':
          absltest.TestCase().assertEqual(x, y)
        else:
          absltest.TestCase().assertEqual(x.battery_charge, y.battery_charge)
  # pylint: enable=protected-access


def create_wind_field() -> wind_field.SimpleStaticWindField:
  """Creates a static wind field to test things in."""
  wf = wind_field.SimpleStaticWindField()
  wf.reset(jax.random.PRNGKey(int(4753849)), units.datetime(1997, 5, 11))

  return wf
