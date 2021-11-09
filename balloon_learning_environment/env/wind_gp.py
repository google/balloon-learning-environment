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

"""Class for summarizing wind observations and forecast.

We use a Gaussian Process to integrate wind observations to a basic forecast.
This lets us query any point (x, y, p, t) in the wind field for its value,
as well as the model confidence's in this value.

---- Open issues
* The forecast is not used

"""

import datetime as dt
from typing import Tuple

from balloon_learning_environment.env import wind_field
from balloon_learning_environment.utils import units
import numpy as np
from sklearn import gaussian_process


_DISTANCE_SCALING = 357000  # [m]
_PRESSURE_SCALING = 326.0  # [Pa]
_TIME_SCALING = 34560  # [seconds]

_SIGMA_EXP_SQUARED = 3.6**2
_SIGMA_NOISE_SQUARED = 0.05


class WindGP(object):
  """Wrapper around a Gaussian Process that handles wind measurements.

  This object models deviations from the forecast ("errors") using a Gaussian
  process over the 4-dimensional space (x, y, pressure, time).

  New measurements are integrated into the GP. Queries return the GP's
  prediction regarding particular 4D location's wind in u, v format, plus
  the GP's confidence about that wind.
  """

  def __init__(self, forecast: wind_field.WindField) -> None:
    """Constructor for the WindGP.

    TODO(bellemare): Currently a forecast is required. This simplifies the
    code somewhat. Whether we keep this depends on a design choice: is a new
    environment built up and torn down per episode, or do we instead use
    reset() functions to reuse objects?

    Args:
      forecast: the forecast wind field.
    """
    self.time_horizon = 6 * 3600  # 6 hours.

    # TODO(bellemare): Add some documentation.
    # TODO(bellemare): I believe this is correct but needs to be validated.
    # The WindGP kernel is a Matern kernel.
    # This rescales the inputs (or equivalently, the distance) by the given
    # scaling factors.
    length_scale = np.array([
        _DISTANCE_SCALING, _DISTANCE_SCALING, _PRESSURE_SCALING, _TIME_SCALING])
    self.kernel = _SIGMA_EXP_SQUARED * gaussian_process.kernels.Matern(
        length_scale=length_scale, length_scale_bounds='fixed', nu=0.5)

    self.model = gaussian_process.GaussianProcessRegressor(
        kernel=self.kernel,  # Matern kernel.
        alpha=_SIGMA_NOISE_SQUARED,  # Add a term to the diagonal of the kernel.
        optimizer=None,  # No optimization.
        )
    self.reset(forecast=forecast)

  def reset(self, forecast: wind_field.WindField) -> None:
    """Resets the the WindGP, effectively erasing previous measurements.

    Args:
      forecast: a 4D forecast for the entire wind field.

    """
    # Erase measurements. Since scikit's GP is all runtime, this is just
    # clearing the list of points.
    self.measurement_locations = []
    self.error_values = []

    # TODO(bellemare): This may change types, as WindField is the 'true'
    # wind field and we may instead want the 'mean' wind field (pre-wind noise).
    self.wind_forecast = forecast

  def observe(self, x: units.Distance, y: units.Distance, pressure: float,
              elapsed_time: dt.timedelta,
              measurement: wind_field.WindVector) -> None:
    """Adds the given measurement to the Gaussian Process.

    Args:
      x: location of the measurement.
      y: location of the measurement.
      pressure: pressure at the measurement.
      elapsed_time: time of the measurement.
      measurement: The wind measured at the location.
    """
    location = np.array([x.meters, y.meters, pressure,
                         elapsed_time.total_seconds()])

    forecast = self.wind_forecast.get_forecast(x, y, pressure, elapsed_time)

    error = np.array([(measurement.u - forecast.u).meters_per_second,
                      (measurement.v - forecast.v).meters_per_second])

    self.measurement_locations.append(location)
    self.error_values.append(error)

  def query(self, x: units.Distance, y: units.Distance, pressure: float,
            elapsed_time: dt.timedelta) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the GP's wind prediction at the given location.

    Args:
      x: the x query coordinate.
      y: the y query coordinate.
      pressure: the query pressure.
      elapsed_time: the time at which to query the GP.

    Returns:
      u: the mean wind direction in x.
      v: the mean wind direction in y.
      confidence: the GP's confidence in these values.
    """
    query_as_array = np.array(
        [[x.meters, y.meters, pressure, elapsed_time.total_seconds()]])
    outputs = self.query_batch(query_as_array)

    # Remove the batch dimension.
    return outputs[0][0], outputs[1][0]

  def query_batch(self, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the GP's wind prediction for a batch of queries.

    Args:
      locations: a N x 4 dimensional array of queries. Each row contains the
        (x, y, p, t) coordinates for one query.
        The 't' argument should be in seconds.
    Returns:
      means: a N x 2 dimensional array. Each row contains the mean wind
        direction (x and y).
      confidence: a N-dim vector. Each element contains the model's uncertainty.

    Raises:
      RuntimeError: if no forecast was previously given.
    """
    # Set up data for the GP.
    # TODO(bellemare): Clearly wasteful if performing multiple queries per
    # observation. Should cache. Premature optimization is the root, etc.
    if not self.measurement_locations:
      means = np.zeros((locations.shape[0], 2))
      deviations = np.zeros(locations.shape[0])
    else:
      if len(self.measurement_locations) == 1:
        # Needed because hstack will leave the dims unchanged with a single
        # row.
        inputs = np.expand_dims(self.measurement_locations[0], axis=0)
        targets = np.expand_dims(self.error_values[0], axis=0)
      else:
        inputs = np.vstack(self.measurement_locations)
        targets = np.vstack(self.error_values)

      # Drop any observations that are more than N hours old. This speeds up
      # computation. Only if all queries have the same time.
      # TODO(bellemare): A slightly more efficient alternative is to drop the
      # data permanently, but this method has the advantage of supporting
      # queries into the past.
      if np.all(locations[:, -1] == locations[0, -1]):
        current_time = locations[0, -1]
        fresh_observations = (
            np.abs(inputs[:, -1] - current_time) < self.time_horizon)

        inputs = inputs[fresh_observations]
        targets = targets[fresh_observations]
      self.model.fit(inputs, targets)

      # Output should be a N x 2 set of predictions about local measurements,
      # and a N-sized vector of standard deviations.
      # TODO(bellemare): Determine why deviations is a single number per sample,
      # instead of two (since we have two values being predicted).
      means, deviations = self.model.predict(locations, return_std=True)

      # Deviations are std.dev., convert to variance and normalize.
      # TODO(bellemare): Ask what the actual lower bound is supposed to
      # be. We can't have a 0 std.dev. due to noise. Currently it's something
      # like 0.07 from the GP, but that doesn't seem to match the Loon code.
      deviations = deviations**2 / _SIGMA_EXP_SQUARED

      # TODO(bellemare): Sal says this needs normalizing so that the lower bound
      # is really zero.

    assert len(means.shape) == 2, means.shape[1] == 2

    self._add_forecast_to_prediction(locations, means)
    return means, deviations

  def _add_forecast_to_prediction(
      self, locations: np.ndarray, means: np.ndarray) -> None:
    """Adds the forecast back to the error prediction.

    The WindGP predicts the error from the forecasts. When that is done, we
    need to recombine it with the forecast to obtain the actual prediction.

    The 'means' vector is modified in-place.

    Args:
      locations: N x 4 array of locations at which predictions have been made.
      means: 2D array of predicted deviations from the forecast.
    """
    # This checks that all x, y, and time are the same in each row.
    assert (locations[1:, [0, 1, 3]] == locations[0, [0, 1, 3]]).all()

    forecasts = self.wind_forecast.get_forecast_column(
        units.Distance(m=locations[0, 0]),
        units.Distance(m=locations[0, 1]),
        locations[:, 2],
        dt.timedelta(seconds=locations[0, 3]))

    for index, forecast in enumerate(forecasts):
      means[index][0] += forecast.u.meters_per_second
      means[index][1] += forecast.v.meters_per_second
