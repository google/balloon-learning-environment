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

"""Tests for balloon_learning_environment.env.standard_atmosphere."""

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.utils import units
import jax


class StandardAtmosphereTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))

  def test_lapse_rates_always_initialized(self):
    # We verify that no errors are thrown.
    self.atmosphere.at_height(units.Distance(meters=10.0))
    self.atmosphere.at_pressure(1000.0)

  @parameterized.named_parameters(
      dict(testcase_name='height_too_low', height=-611.0),
      dict(testcase_name='height_too_high', height=85000.0))
  def test_at_height_edge_cases(self, height):
    with self.assertRaises(AssertionError):
      self.atmosphere.at_height(units.Distance(meters=height))

  @parameterized.named_parameters(
      dict(testcase_name='pressure_too_low', pressure=0.0),
      dict(testcase_name='pressure_too_high', pressure=110_000.0))
  def test_at_pressure_edge_cases(self, pressure):
    with self.assertRaises(AssertionError):
      self.atmosphere.at_pressure(pressure)

  @parameterized.parameters(
      (-610.0, (108870.81, 108870.83), (299.99, 300.01), (1.25, 1.27)),
      (0.0, (101515.13, 101522.11), (295.81, 296.41), (1.18, 1.21)),
      (1000.0, (90320.09, 90364.48), (288.95, 290.50), (1.07, 1.10)),
      (5000.0, (54952.29, 55323.66), (261.52, 266.88), (0.71, 0.74)),
      (11000.0, (23422.04, 24266.60), (220.37, 231.44),
       (0.36, 0.38)), (15000.0, (12078.87, 13017.38), (192.93, 207.82),
                       (0.21, 0.23)), (18000.0, (6934.84, 7812.34),
                                       (185.10, 201.10), (0.12, 0.15)),
      (20000.0, (4848.63, 5608.26), (196.86, 211.27),
       (0.08, 0.10)), (22000.0, (3454.32, 4083.31), (203.74, 217.36),
                       (0.05, 0.08)), (25000.0, (2096.53, 2556.40),
                                       (206.74, 220.36), (0.03, 0.05)),
      (30000.0, (926.75, 1187.77), (211.74, 225.36),
       (0.01, 0.03)), (32000.0, (672.18, 878.28), (213.74, 227.36),
                       (0.00, 0.02)), (35000.0, (419.99, 564.14),
                                       (222.14, 235.76), (-0.00, 0.02)),
      (40000.0, (199.25, 279.07), (236.14, 249.76),
       (-0.01, 0.01)), (45000.0, (98.67, 143.46), (250.14, 263.76),
                        (-0.01, 0.01)), (47000.0, (75.31, 111.02),
                                         (255.74, 269.36), (-0.01, 0.01)),
      (50000.0, (50.44, 75.89), (255.74, 269.36),
       (-0.01, 0.01)), (51000.0, (44.13, 66.85), (255.74, 269.36),
                        (-0.01, 0.01)), (55000.0, (25.55, 39.82),
                                         (244.54, 258.16), (-0.01, 0.01)),
      (60000.0, (12.44, 20.17), (230.54, 244.16),
       (-0.01, 0.01)), (65000.0, (5.79, 9.82), (216.54, 230.16), (-0.01, 0.01)),
      (70000.0, (2.56, 4.57), (202.54, 216.16),
       (-0.01, 0.01)), (71000.0, (2.15, 3.90), (199.74, 213.36), (-0.01, 0.01)),
      (75000.0, (1.07, 2.04), (191.74, 205.36),
       (-0.01, 0.01)), (80000.0, (0.42, 0.87), (181.74, 195.36),
                        (-0.01, 0.01)), (84500.0, (0.17, 0.40),
                                         (172.74, 186.36), (-0.01, 0.01)))
  def test_at_height_with_sane_data(self, height, pressure_range,
                                    temperature_range, density_range):
    for key in range(10):
      self.atmosphere.reset(jax.random.PRNGKey(key))
      atm = self.atmosphere.at_height(units.Distance(meters=height))
      self.assertBetween(atm.pressure, *pressure_range)
      self.assertBetween(atm.temperature, *temperature_range)
      self.assertBetween(atm.density, *density_range)

  @parameterized.parameters(
      (108870, (-609.94, -609.92), (299.99, 300.01), (1.25, 1.27)),
      (101325, (16.22, 16.86), (295.70, 296.31), (1.18, 1.20)),
      (89874, (1041.85, 1046.26), (288.66, 290.23), (1.07, 1.09)),
      (54019, (5130.90, 5186.04), (260.62, 265.78), (0.70, 0.73)),
      (22632, (11220.57, 11469.59), (218.85, 228.67), (0.33, 0.37)),
      (12044, (15016.32, 15469.60), (192.82, 205.05), (0.19, 0.23)),
      (7504.8, (17574.93, 18237.11), (182.60, 202.30), (0.12, 0.15)),
      (5474.8, (19307.33, 20149.21), (192.79, 212.03), (0.08, 0.11)),
      (3999.7, (21127.58, 22131.65), (202.87, 217.49), (0.05, 0.08)),
      (2511.0, (23911.14, 25115.57), (205.66, 220.48), (0.03, 0.05)),
      (1171.8, (28550.85, 30089.24), (210.30, 225.45), (0.01, 0.03)),
      (868.01, (30406.32, 32078.29), (212.15, 227.58), (0.00, 0.02)),
      (558.92, (33163.38, 35064.11), (217.00, 235.94), (-0.00, 0.02)),
      (277.52, (37740.71, 40040.53), (229.82, 249.88), (-0.01, 0.01)),
      (143.13, (42318.24, 45017.16), (242.64, 263.81), (-0.01, 0.01)),
      (110.9, (44149.35, 47007.92), (247.76, 269.36), (-0.01, 0.01)),
      (75.944, (46938.47, 49993.17), (255.57, 269.36), (-0.01, 0.01)),
      (66.938, (47883.43, 50988.40), (255.74, 269.36), (-0.01, 0.01)),
      (39.969, (51740.79, 54969.80), (253.67, 258.25), (-0.01, 0.01)),
      (20.314, (56629.50, 59946.34), (239.98, 244.31), (-0.01, 0.01)),
      (9.922, (61518.23, 64922.91), (226.29, 230.38), (-0.01, 0.01)),
      (4.6342, (66407.05, 69899.55), (212.60, 216.44), (-0.01, 0.01)),
      (3.9564, (67384.81, 70894.88), (209.87, 213.66), (-0.01, 0.01)),
      (2.0679, (71266.65, 74874.66), (199.21, 205.61), (-0.01, 0.01)),
      (0.88627, (76086.91, 79849.35), (189.57, 195.66), (-0.01, 0.01)),
      (0.39814, (80425.11, 84326.54), (180.89, 186.71), (-0.01, 0.01)))
  def test_at_pressure_with_sane_data(self, pressure, height_range,
                                      temperature_range, density_range):
    for key in range(10):
      self.atmosphere.reset(jax.random.PRNGKey(key))
      atm = self.atmosphere.at_pressure(pressure)
      self.assertBetween(atm.height.m, *height_range)
      self.assertBetween(atm.temperature, *temperature_range)
      self.assertBetween(atm.density, *density_range)


if __name__ == '__main__':
  absltest.main()
