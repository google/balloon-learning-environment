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

"""Tests for balloon_learning_environment.env.wind_gp."""

import datetime as dt

from absl.testing import absltest
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env import wind_gp
from balloon_learning_environment.utils import units


class WindGpTest(absltest.TestCase):

  def setUp(self):
    super(WindGpTest, self).setUp()
    # TODO(bellemare): If multiple wind fields are available, use the
    # vanilla (4 sheets) one.
    wf = wind_field.SimpleStaticWindField()

    # Sets up a model with a dummy forecast.
    model = wind_gp.WindGP(wf)
    self.model = model
    self.x = units.Distance(m=0.0)
    self.y = units.Distance(m=0.0)
    self.pressure = 0.0
    self.delta = dt.timedelta(seconds=0)
    self.wind_vector = wind_field.WindVector(
        units.Velocity(mps=1.0), units.Velocity(mps=1.0))

  def test_measurement_has_almost_no_variance(self):
    self.model.observe(self.x, self.y, self.pressure, self.delta,
                       self.wind_vector)
    post_measurement = self.model.query(
        self.x, self.y, self.pressure, self.delta)

    # Variance is measurement noise at a measured point.
    # This constant was computed anaytically. It is given by (see wind_gp.py):
    #   SIGMA_NOISE_SQUARED / (SIGMA_NOISE_SQUARED + SIGMA_EXP_SQUARED) .
    self.assertAlmostEqual(post_measurement[1].item(), 0.003843, places=3)

  def test_observations_affect_forecast_continuously(self):
    pre_measurement = self.model.query(
        self.x, self.y, self.pressure, self.delta)
    self.model.observe(self.x, self.y, self.pressure, self.delta,
                       self.wind_vector)
    post_measurement = self.model.query(
        units.Distance(km=0.05), self.y, self.pressure, self.delta)

    self.assertTrue((pre_measurement[0] != post_measurement[0]).all())

if __name__ == '__main__':
  absltest.main()
