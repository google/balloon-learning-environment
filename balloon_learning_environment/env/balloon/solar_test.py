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

"""Tests for solar.py."""

import datetime as dt
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.utils import units

import s2sphere as s2


# For brevity.
def s2latlng(lat: float, lng: float) -> s2.LatLng:
  return s2.LatLng.from_degrees(lat, lng)


class SolarTest(parameterized.TestCase):

  def setUp(self):
    super(SolarTest, self).setUp()
    self._dt = units.datetime(2012, 6, 1)

  @parameterized.named_parameters(
      dict(testcase_name='LatTooHigh', latlng=s2latlng(91.0, 0.0)),
      dict(testcase_name='LatTooLow', latlng=s2latlng(-91.0, 0.0)))
  def testSolarCalculatorRaisesWithInvalidLatitude(self, latlng: s2.LatLng):
    with self.assertRaises(ValueError):
      _ = solar.solar_calculator(latlng, self._dt)

  @parameterized.named_parameters(
      dict(
          testcase_name='1',
          latlng=s2latlng(37.3894, -122.0819),
          timestamp=1382123322,
          expected_el_degree=41.6,
          expected_az_degree=None,
          expected_flux=None),
      dict(
          testcase_name='2',
          latlng=s2latlng(37.3861, -122.0828),
          timestamp=1374188880,
          expected_el_degree=49.14,
          expected_az_degree=258.56,
          expected_flux=1320.16),
      dict(
          testcase_name='3',
          latlng=s2latlng(-35.1234, -71.5720),
          timestamp=1367743680,
          expected_el_degree=-32.63,
          expected_az_degree=92.41,
          expected_flux=1342.24),
      dict(
          testcase_name='4',
          latlng=s2latlng(-70.0, -105.0),
          timestamp=1358237160,
          expected_el_degree=1.93,
          expected_az_degree=166.82,
          expected_flux=1412.20),
      dict(
          testcase_name='5',
          latlng=s2latlng(0.0, 0.0),
          timestamp=1357041600,
          expected_el_degree=67.03,
          expected_az_degree=177.84,
          expected_flux=1413.17),
      dict(
          testcase_name='6',
          latlng=s2latlng(0.0, 180.0),
          timestamp=1357041600,
          expected_el_degree=-67.02,
          expected_az_degree=182.16,
          expected_flux=1413.17),
  )
  def testSolarCalculatorWorks(self, latlng: s2.LatLng, timestamp: int,
                               expected_el_degree: float,
                               expected_az_degree: Optional[float],
                               expected_flux: Optional[float]):
    el_deg, az_degree, flux = solar.solar_calculator(
        latlng, units.datetime_from_timestamp(timestamp))

    self.assertAlmostEqual(el_deg, expected_el_degree, places=1)
    if expected_az_degree is not None:
      self.assertAlmostEqual(az_degree, expected_az_degree, places=1)
    if expected_flux is not None:
      self.assertAlmostEqual(flux, expected_flux, places=1)

  @parameterized.named_parameters(
      dict(testcase_name='ElDegreeTooHigh', el_degree=91.0),
      dict(testcase_name='ElDegreeTooLow', el_degree=-91.0))
  def testSolarAtmosphericAttenuationRaisesWithInvalidElevation(
      self, el_degree: float):
    with self.assertRaises(ValueError):
      _ = solar.solar_atmospheric_attenuation(el_degree, 5000.0)

  @parameterized.named_parameters(
      dict(testcase_name='PressureTooLow', pressure=-1.0),
      dict(testcase_name='PressureTooHigh', pressure=101326.0))
  def testSolarAtmosphericAttenuationRaisesWithInvalidPressure(
      self, pressure: float):
    with self.assertRaises(ValueError):
      _ = solar.solar_atmospheric_attenuation(0.0, pressure)

  @parameterized.named_parameters(
      dict(
          testcase_name='1_0',
          el_deg=0.0,
          pressure_altitude_pa=101325.0,
          expected=0.000186),
      dict(
          testcase_name='1_30',
          el_deg=30.0,
          pressure_altitude_pa=101325.0,
          expected=0.577255),
      dict(
          testcase_name='1_60',
          el_deg=60.0,
          pressure_altitude_pa=101325.0,
          expected=0.726702),
      dict(
          testcase_name='1_90',
          el_deg=90.0,
          pressure_altitude_pa=101325.0,
          expected=0.758242),
      dict(
          testcase_name='2_0',
          el_deg=0.0,
          pressure_altitude_pa=20000.0,
          expected=0.155560),
      dict(
          testcase_name='2_30',
          el_deg=30.0,
          pressure_altitude_pa=20000.0,
          expected=0.896450),
      dict(
          testcase_name='2_60',
          el_deg=60.0,
          pressure_altitude_pa=20000.0,
          expected=0.938662),
      dict(
          testcase_name='2_90',
          el_deg=90.0,
          pressure_altitude_pa=20000.0,
          expected=0.946635),
      dict(
          testcase_name='3_0',
          el_deg=0.0,
          pressure_altitude_pa=5000.0,
          expected=0.620610),
      dict(
          testcase_name='3_30',
          el_deg=30.0,
          pressure_altitude_pa=5000.0,
          expected=0.973003),
      dict(
          testcase_name='3_60',
          el_deg=60.0,
          pressure_altitude_pa=5000.0,
          expected=0.984287),
      dict(
          testcase_name='3_90',
          el_deg=90.0,
          pressure_altitude_pa=5000.0,
          expected=0.986373),
      dict(
          testcase_name='4_00',
          el_deg=90.0,
          pressure_altitude_pa=0.0,
          expected=1.0),
      dict(
          testcase_name='4_30',
          el_deg=90.0,
          pressure_altitude_pa=0.0,
          expected=1.0))
  def testSolarAtmosphericAttenuationWorks(self, el_deg: float,
                                           pressure_altitude_pa: float,
                                           expected: float):
    self.assertAlmostEqual(
        solar.solar_atmospheric_attenuation(el_deg, pressure_altitude_pa),
        expected, places=5)

  @parameterized.named_parameters(
      dict(testcase_name='3m1', el_deg=90.0, panel_height=3.0, expected=0.4392),
      dict(testcase_name='3m2', el_deg=45.0, panel_height=3.0, expected=0.4392),
      dict(testcase_name='3m3', el_deg=30.0, panel_height=3.0, expected=1.0),
      dict(testcase_name='3m4', el_deg=0.0, panel_height=3.0, expected=1.0),
      dict(testcase_name='1m1', el_deg=90.0, panel_height=1.0, expected=0.4392),
      dict(testcase_name='1m2', el_deg=45.0, panel_height=1.0, expected=0.4392),
      dict(testcase_name='1m3', el_deg=30.0, panel_height=1.0, expected=0.4392),
      dict(testcase_name='1m4', el_deg=0.0, panel_height=1.0, expected=1.0))
  def testBalloonShadowWorks(
      self, el_deg: float, panel_height: float, expected: float):
    self.assertAlmostEqual(
        solar.balloon_shadow(el_deg, panel_height), expected, places=3)

  @parameterized.named_parameters(
      dict(
          testcase_name='9am',
          datetime=units.datetime(2013, 9, 21, 9),
          expected_sunrise=units.datetime(2013, 9, 22, 5, 36),
          expected_sunset=units.datetime(2013, 9, 21, 18, 9)),
      dict(
          testcase_name='3pm',
          datetime=units.datetime(2013, 9, 21, 15),
          expected_sunrise=units.datetime(2013, 9, 22, 5, 36),
          expected_sunset=units.datetime(2013, 9, 21, 18, 9)),
      dict(
          testcase_name='9pm',
          datetime=units.datetime(2013, 9, 21, 21),
          expected_sunrise=units.datetime(2013, 9, 22, 5, 36),
          expected_sunset=units.datetime(2013, 9, 22, 18, 9)),
      dict(
          testcase_name='3am',
          datetime=units.datetime(2013, 9, 22, 3),
          expected_sunrise=units.datetime(2013, 9, 22, 5, 36),
          expected_sunset=units.datetime(2013, 9, 22, 18, 9)))
  def testSunriseSunsetCalculation(self, datetime: dt.datetime,
                                   expected_sunrise: dt.datetime,
                                   expected_sunset: dt.datetime):
    # When is the next sunrise and sunset from 9am on Sept. 21st?
    sunrise, sunset = solar.get_next_sunrise_sunset(
        s2latlng(0.0, 0.0), datetime)
    # Technically, this output depends on the resolution at which we find
    # sunset/sunrise, so the test will fail if the default timeDelta is changed.
    self.assertEqual(sunrise, expected_sunrise)
    self.assertEqual(sunset, expected_sunset)

  @parameterized.named_parameters(
      dict(
          testcase_name='Daytime',
          datetime=units.datetime(2013, 9, 21, 12),
          expected=0),
      dict(
          testcase_name='AlmostSunrise',
          datetime=units.datetime(2013, 9, 21, 5, 30),
          expected=2))
  def testStepsToSunrise(self, datetime: dt.datetime, expected: int):
    steps_to_sunrise = solar.calculate_steps_to_sunrise(
        s2latlng(0.0, 0.0), datetime)
    self.assertEqual(steps_to_sunrise, expected)

if __name__ == '__main__':
  absltest.main()
