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

"""ACS power as a function of superpressure."""

from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import units
import numpy as np
from scipy import interpolate


_PRESSURE_RATIO_TO_POWER: interpolate.interpolate.interp1d = (
    interpolate.interp1d(
        np.array([1.0, 1.05, 1.2, 1.25, 1.35]),  # pressure_ratio
        np.array([100.0, 100.0, 300.0, 400.0, 400.0]),  # power
        fill_value='extrapolate'))


_PRESSURE_RATIO_POWER_TO_EFFICIENCY: interpolate.interpolate.interp2d = (
    interpolate.interp2d(
        np.linspace(1.05, 1.35, 13),  # pressure_ratio
        np.linspace(100.0, 400.0, 4),  # power
        np.array([0.4, 0.4, 0.3, 0.2, 0.2, 0.00000, 0.00000, 0.00000, 0.00000,
                  0.00000, 0.00000, 0.00000, 0.00000, 0.4, 0.3, 0.3, 0.30, 0.25,
                  0.23, 0.20, 0.15, 0.12, 0.10, 0.00000, 0.00000, 0.00000,
                  0.00000, 0.3, 0.25, 0.25, 0.25, 0.20, 0.20, 0.20, 0.2, 0.15,
                  0.13, 0.12, 0.11, 0.00000, 0.23, 0.23, 0.23, 0.23, 0.23, 0.20,
                  0.20, 0.20, 0.18, 0.16, 0.15, 0.13]),  # efficiency
        fill_value=None))


def get_most_efficient_power(pressure_ratio: float) -> units.Power:
  """Lookup the optimal operating power from static tables.

  Gets the optimal operating power [W] in terms of kg of air moved per unit
  energy.

  Args:
    pressure_ratio: Ratio of (balloon pressure + superpresure) to baloon
      pressure.

  Returns:
    Optimal ACS power at current pressure_ratio.
  """
  power = _PRESSURE_RATIO_TO_POWER(pressure_ratio)
  return units.Power(watts=power)


def get_fan_efficiency(pressure_ratio: float, power: units.Power) -> float:
  # Compute efficiency of air flow from current pressure ratio and power.
  efficiency = _PRESSURE_RATIO_POWER_TO_EFFICIENCY(pressure_ratio, power.watts)
  return float(efficiency)


def get_mass_flow(power: units.Power, efficiency: float) -> float:
  return efficiency * power.watts / constants.NUM_SECONDS_PER_HOUR
