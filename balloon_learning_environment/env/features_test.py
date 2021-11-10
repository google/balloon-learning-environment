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

"""Tests for balloon_learning_environment.env.features."""

import datetime as dt
import itertools
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env import features
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env.balloon import altitude_safety
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import test_helpers
from balloon_learning_environment.utils import units
import jax
import numpy as np

START_DATETIME = units.datetime(2013, 3, 25, 9, 25, 32)


class PerciatelliFeaturesTest(parameterized.TestCase):

  def setUp(self):
    super(PerciatelliFeaturesTest, self).setUp()
    self.invalid_vector = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    # Sets up a model with a dummy forecast.
    self.wf = test_helpers.create_wind_field()
    self.atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(38))
    self.features = features.PerciatelliFeatureConstructor(
        self.wf, self.atmosphere)

  def create_observation(
      self,
      pressure: float = 9000.0,
      charge_percent: float = 1.0,
      x_km: float = 0.0,
      y_km: float = 0.0,
      lat: float = 0.0,
      lng: float = 0.0,
      last_command: control.AltitudeControlCommand = (
          control.AltitudeControlCommand.STAY),
      datetime: dt.datetime = START_DATETIME,
      navigation_is_paused: bool = False
  ) -> simulator_data.SimulatorObservation:
    # TODO(joshgreaves): power_percent or charge_percent?
    b = test_helpers.create_balloon(
        pressure=pressure,
        power_percent=charge_percent,
        x=units.Distance(km=x_km),
        y=units.Distance(km=y_km),
        center_lat=lat,
        center_lng=lng,
        date_time=datetime,
        atmosphere=self.atmosphere)
    balloon_state = b.state
    balloon_state.last_command = last_command
    type(balloon_state).navigation_is_paused = mock.PropertyMock(
        return_value=navigation_is_paused)

    # Get the wind that matches the forecast. We may need to update this in the
    # future for testing the WindGP. Also note, this object is very lightweight
    # so it is fine to create it here.
    wf = test_helpers.create_wind_field()

    # Here we're using the forecast as a ground truth wind, to avoid
    # nondeterminism due to wind noise.
    wv = wf.get_forecast(
        units.Distance(km=0.0), units.Distance(km=0.0), pressure,
        dt.timedelta(seconds=0))

    return simulator_data.SimulatorObservation(balloon_state, wv)

  def test_make_features(self):
    """Verify that the get_features method returns the right kind of object."""
    observation = self.create_observation()

    self.features.observe(observation)
    vector = self.features.get_features()

    self.assertIsInstance(vector, np.ndarray)
    self.assertEqual(vector.shape, (1099,))

  # --------------- Wind Column Tests ---------------

  def test_invalid_range_is_correctly_padded(self):
    """Verify that get_features produces the right padding."""
    observation = self.create_observation()

    self.features.observe(observation)
    vector = self.features.get_features()
    wind_column = vector[16:]  # Remove ambient features.

    # The balloon should be at level 80, which corresponds to [0, 100] and
    # [281, 361] being invalid.
    for i in itertools.chain(range(100), range(281, 361)):
      # Use tuple equality to keep things simple.
      self.assertEqual(
          tuple(wind_column[3 * i: 3 * (i + 1)]),
          tuple(self.invalid_vector),
          msg='Non-invalid wind at index {}'.format(i))

  def test_valid_winds(self):
    """Verify that get_features produces the right padding."""
    observation = self.create_observation()

    self.features.observe(observation)
    vector = self.features.get_features()
    wind_column = vector[16:]  # Remove ambient features.

    # This test relies on the fact that SimpleStaticWindField should return
    # a forecast that is exactly 10 m/s in some direction.
    # Below 130 corresponds to higher altitudes than the balloon can reach.
    # Above 231 corresponds to lower altitudes than we allow.
    for i in range(130, 232):
      magnitude = wind_column[3 * i + 2]
      # Magnitude 0.25 should correspond to 10 m/s after squash (10 / (10+30)).
      self.assertEqual(magnitude, 0.25, msg=f'Bad wind at index {i}')

  def test_low_pressure_balloon_creates_valid_wind_features(self):
    observation = self.create_observation(
        constants.PERCIATELLI_PRESSURE_RANGE_MIN)

    self.features.observe(observation)
    vector = self.features.get_features()
    wind_column = vector[16:]  # Remove ambient features.

    # The balloon should be at level 0, which corresponds to [0, 179]
    # being invalid.
    invalid_range = wind_column[:3 * 179].reshape(-1, 3)
    self.assertTrue((invalid_range == self.invalid_vector).all())

  def test_high_pressure_balloon_creates_valid_wind_features(self):
    observation = self.create_observation(
        constants.PERCIATELLI_PRESSURE_RANGE_MAX)

    self.features.observe(observation)
    vector = self.features.get_features()
    wind_column = vector[16:]  # Remove ambient features.

    # The balloon should be at level 181, which corresponds to [181, 361]
    # being invalid.
    invalid_range = wind_column[181 * 3:361 * 3].reshape(-1, 3)
    self.assertTrue((invalid_range == self.invalid_vector).all())

  def test_unreachable_altitude_is_marked_as_uncreachable(self):
    # Set balloon to the minimum altitude.
    pressure = self.atmosphere.at_height(altitude_safety.MIN_ALTITUDE).pressure
    observation = self.create_observation(pressure=pressure)

    self.features.observe(observation)
    vector = self.features.get_features()
    wind_column = vector[16:].reshape(-1, 3)  # Remove ambient features.

    # The balloon's altitude should be reachable.
    # Balloon is always found at pressure level 180.
    pressure_level = 180
    valid_feature = wind_column[pressure_level]
    self.assertFalse((valid_feature == self.invalid_vector).all())

    # Everything under the balloon should be unreachable.
    invalid_range = wind_column[pressure_level + 1:]
    self.assertTrue((invalid_range == self.invalid_vector).all())

  # --------------- Ambient Feature Tests ---------------

  # TODO(joshgreaves): Submit after Sal to get
  # PERCIATELLI_PRESSURE_RANGE_MIN/MAX.
  @parameterized.named_parameters(
      dict(
          testcase_name='max_pressure',
          pressure=constants.PERCIATELLI_PRESSURE_RANGE_MAX,
          expected=1.0),
      dict(
          testcase_name='over_max_pressure',
          pressure=constants.PERCIATELLI_PRESSURE_RANGE_MAX + 100.0,
          expected=1.0),
      dict(
          testcase_name='min_pressure',
          pressure=constants.PERCIATELLI_PRESSURE_RANGE_MIN,
          expected=0.0),
      dict(
          testcase_name='under_min_pressure',
          pressure=constants.PERCIATELLI_PRESSURE_RANGE_MIN - 100.0,
          expected=0.0),
      dict(
          testcase_name='mid_pressure',
          pressure=(
              (constants.PERCIATELLI_PRESSURE_RANGE_MIN +
               constants.PERCIATELLI_PRESSURE_RANGE_MAX)
              / 2.0),
          expected=0.5))
  def test_pressure_ambient_feature_correctly_set(self, pressure: float,
                                                  expected: float):
    # 1st ambient feature, found at feature_vector[0].
    observation = self.create_observation(pressure=pressure)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertAlmostEqual(feature_vector[0], expected)

  @parameterized.named_parameters(
      dict(testcase_name='full_power', charge_percent=1.0),
      dict(testcase_name='no_power', charge_percent=0.0),
      dict(testcase_name='some_power', charge_percent=0.32))
  def test_power_ambient_feature_correctly_set(self, charge_percent: float):
    # 2nd ambient feature, found at feature_vector[1].
    observation = self.create_observation(charge_percent=charge_percent)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertAlmostEqual(feature_vector[1], charge_percent)

  # Locations and datetimes from NOAA solar calculator.
  # https://gml.noaa.gov/grad/solcalc/
  @parameterized.named_parameters(
      dict(
          testcase_name='high_noon_vernal_equinox',
          lat=0.0,
          lng=0.0,
          datetime=units.datetime(2022, 3, 20, 12, 7, 27),
          low=0.99,
          high=1.0),
      dict(
          testcase_name='midnight_vernal_equinox',
          lat=0.0,
          lng=180.0,
          datetime=units.datetime(2022, 3, 20, 12, 7, 27),
          low=0.0,
          high=0.01),
      dict(
          testcase_name='morning_vernal_equinox',
          lat=0.0,
          lng=0.0,
          datetime=units.datetime(2022, 3, 20, 8, 32, 12),
          low=0.6,
          high=0.9))
  def test_solar_angle_ambient_feature_correctly_set(self, lat: float,
                                                     lng: float,
                                                     datetime: dt.datetime,
                                                     low: float, high: float):
    # 3rd ambient feature, found at feature_vector[2].
    # TODO(joshgreaves): Make a self.create_observation function.
    observation = self.create_observation(lat=lat, lng=lng, datetime=datetime)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertBetween(feature_vector[2], low, high)

  @unittest.skip('Unsure how to write this test.')
  def test_sin_normalized_solar_cycle_ambient_feature_set_correctly(self):
    # 4th ambient feature, found at feature_vector[3].
    # TODO(joshgreaves): Write this test.
    pass

  @unittest.skip('Unsure how to write this test.')
  def test_cos_normalized_solar_cycle_ambient_feature_set_correctly(self):
    # 5th ambient feature, found at feature_vector[4].
    # TODO(joshgreaves): Write this test.
    pass

  @parameterized.named_parameters(
      dict(testcase_name='west', x_km=1.0, y_km=0.0, expected=-1.0),
      dict(testcase_name='east', x_km=-1.0, y_km=0.0, expected=1.0),
      dict(testcase_name='north', x_km=0.0, y_km=-1.0, expected=0.0),
      dict(testcase_name='south', x_km=0.0, y_km=1.0, expected=0.0))
  def test_sin_heading_to_target_ambient_feature_set_correctly(
      self, x_km: float, y_km: float, expected: float):
    # 6th ambient feature, found at feature_vector[5].
    observation = self.create_observation(x_km=x_km, y_km=y_km)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertAlmostEqual(feature_vector[5], expected)

  @parameterized.named_parameters(
      dict(testcase_name='west', x_km=1.0, y_km=0.0, expected=0.0),
      dict(testcase_name='east', x_km=-1.0, y_km=0.0, expected=0.0),
      dict(testcase_name='north', x_km=0.0, y_km=-1.0, expected=1.0),
      dict(testcase_name='south', x_km=0.0, y_km=1.0, expected=-1.0))
  def test_cos_heading_to_target_ambient_feature_set_correctly(
      self, x_km: float, y_km: float, expected: float):
    # 7th ambient feature, found at feature_vector[6].
    observation = self.create_observation(x_km=x_km, y_km=y_km)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertAlmostEqual(feature_vector[6], expected)

  @parameterized.named_parameters(
      dict(testcase_name='south', x_km=0.0, y_km=1.0, expected=1 / 251),
      dict(testcase_name='east', x_km=-3.67, y_km=0.0, expected=3.67 / 253.67),
      dict(testcase_name='far', x_km=300.0, y_km=400.0, expected=2 / 3),
      dict(testcase_name='zero', x_km=0.0, y_km=0.0, expected=0.0))
  def test_distance_to_target_ambient_feature_set_correctly(
      self, x_km: float, y_km: float, expected: float):
    # 8th ambient feature, found at feature_vector[7].
    observation = self.create_observation(x_km=x_km, y_km=y_km)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertAlmostEqual(feature_vector[7], expected)

  @parameterized.named_parameters(
      dict(testcase_name='up', last_command=control.AltitudeControlCommand.UP),
      dict(
          testcase_name='stay',
          last_command=control.AltitudeControlCommand.STAY),
      dict(
          testcase_name='down',
          last_command=control.AltitudeControlCommand.DOWN))
  def test_last_command_ambient_features_set_correctly(
      self, last_command: control.AltitudeControlCommand):
    # 9th, 10th, 11th ambient features. Found at feature_vector[8], [9], [10].
    # In order: ascend, stop, descend.
    observation = self.create_observation(last_command=last_command)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    up = 1.0 if last_command == control.AltitudeControlCommand.UP else 0.0
    stay = 1.0 if last_command == control.AltitudeControlCommand.STAY else 0.0
    down = 1.0 if last_command == control.AltitudeControlCommand.DOWN else 0.0
    self.assertEqual(feature_vector[8], up)
    self.assertEqual(feature_vector[9], stay)
    self.assertEqual(feature_vector[10], down)

  @parameterized.named_parameters(
      dict(testcase_name='is_paused', is_paused=True),
      dict(testcase_name='is_not_paused', is_paused=False))
  def test_navigation_is_paused_ambient_feature_correctly_set(
      self, is_paused: bool):
    # 12th and 13th ambient features. Found at feature_vector[11], [12].
    observation = self.create_observation(navigation_is_paused=is_paused)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertEqual(feature_vector[11], 1.0 if is_paused else 0.0)
    self.assertEqual(feature_vector[12], 1.0 if not is_paused else 0.0)

  @parameterized.named_parameters(
      dict(
          testcase_name='daytime_full_battery',
          datetime=units.datetime(
              year=2020, month=6, day=21, hour=12, minute=0, second=0),
          power_percent=1.0,
          expected=1.0),
      dict(
          testcase_name='daytime_partial_battery',
          datetime=units.datetime(
              year=2020, month=6, day=21, hour=12, minute=0, second=0),
          power_percent=0.5,
          expected=0.0),
      dict(
          testcase_name='nighttime_full_battery',
          datetime=units.datetime(
              year=2020, month=6, day=21, hour=0, minute=0, second=0),
          power_percent=1.0,
          expected=0.0),
      dict(
          testcase_name='nighttime_partial_battery',
          datetime=units.datetime(
              year=2020, month=6, day=21, hour=0, minute=0, second=0),
          power_percent=0.1,
          expected=0.0))
  def test_excess_energy_ambient_feature_correctly_set(
      self, datetime: dt.datetime, power_percent: float, expected: float):
    # 14th ambient feature. Found at feature_vector[13].
    # Assumes latlng (0, 0).
    observation = self.create_observation(
        datetime=datetime, charge_percent=power_percent)

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertEqual(feature_vector[13], expected)

  def test_acs_power_to_use_is_in_unit_interval(self):
    # 15th ambient feature. Found at feature_vector[14].
    observation = self.create_observation()

    self.features.observe(observation)
    feature_vector = self.features.get_features()

    self.assertBetween(feature_vector[14], 0.0, 1.0)

  def test_acs_power_to_use_dereases_at_low_altitude(self):
    # 15th ambient feature. Found at feature_vector[14].
    # The ACS takes energy to pump air into the envelope. The higher the
    # ambient pressure (i.e. the lower the altitude), the easier that is.
    low_observation = self.create_observation(pressure=12_000.0)
    high_observation = self.create_observation(pressure=5_000.0)

    # In this test, it is fine to jump the balloon around, since we are just
    # looking at the acs_power_to_use feature, which doesn't depend on history.
    self.features.observe(low_observation)
    low_fv = self.features.get_features()
    self.features.observe(high_observation)
    high_fv = self.features.get_features()

    self.assertGreater(high_fv[14], low_fv[14])

  # TODO(joshgreaves): Test ACS pressure ratio feature.

  # --------------- Named Feature Tests ---------------

  def test_perciatelli_features_correctly_parses_vector(self):
    observation = self.create_observation()

    self.features.observe(observation)
    vector = self.features.get_features()
    named_features = features.NamedPerciatelliFeatures(vector)

    self.assertAlmostEqual(
        named_features.balloon_pressure,
        observation.balloon_observation.pressure,
        places=4)

    # The balloon should be at level 80, which corresponds to [0, 100] and
    # [281, 361] being invalid.
    for i in itertools.chain(range(100), range(281, 361)):
      self.assertEqual(
          named_features.wind_at(i),
          features.PerciatelliWindFeature(*self.invalid_vector))

  def test_invalid_range_is_parsed_as_such(self):
    """Verify that NamedPerciatelliFeatures correctly parses invalid winds."""
    observation = self.create_observation()
    self.features.observe(observation)
    vector = self.features.get_features()
    named_features = features.NamedPerciatelliFeatures(vector)

    for i in range(361):
      # Below 130 is above max altitude (0 mols_air in ballonet).
      # Above 231 is below min altitude.
      if i < 126 or i > 252:
        self.assertFalse(
            named_features.level_is_valid(i),
            msg='Invalid wind parsed as valid at index {}'.format(i))
      else:
        self.assertTrue(
            named_features.level_is_valid(i),
            msg='Valid wind parsed as invalid at index {}'.format(i))

  # --------------- Other Tests ---------------

  def test_compute_solar_angle(self):
    bal = test_helpers.create_balloon(
        pressure=constants.PERCIATELLI_PRESSURE_RANGE_MIN,
        atmosphere=self.atmosphere).state

    bal.date_time = units.datetime(2013, 9, 21, 12, 0, 0)
    noon_angle = features.compute_solar_angle(bal)
    bal.date_time = units.datetime(2013, 9, 21, 0, 0, 0)
    midnight_angle = features.compute_solar_angle(bal)
    bal.date_time = units.datetime(2013, 9, 21, 18, 0, 0)
    sunset_angle = features.compute_solar_angle(bal)

    self.assertGreater(noon_angle, 80)  # It should be pretty high up.
    self.assertLess(midnight_angle, -80)  # It should be on the other side.
    self.assertAlmostEqual(sunset_angle, -1.57684695242166)

    # TODO(joshgreaves): Add tests for ambient features as they are added.

    # TODO(bellemare): Test that all features lie in [0, 1] with a more
    # interesting wind field.


if __name__ == '__main__':
  absltest.main()
