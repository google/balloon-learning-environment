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

"""A lookup table from pressure ratio, state of charge -> power to use."""

import bisect


def lookup(pressure_ratio: float,
           state_of_charge: float) -> float:
  """Map pressure_ratio x state_of_charge to power to use."""
  assert pressure_ratio >= 0.99 and pressure_ratio <= 5
  pressure_ratio_intervals = [1.08, 1.11, 1.14, 1.17, 1.2, 1.23, 1.26]
  soc_mappings = [  # One entry for each pressure ratio interval.
      ([0.3, 0.4, 0.5], [0, 150, 175, 200]),  # 0.99 -> 1.08
      ([0.3, 0.4, 0.7], [0, 200, 200, 225]),  # 1.08 -> 1.11
      ([0.3, 0.4, 0.6], [0, 225, 225, 250]),  # 1.11 -> 1.14
      ([0.3, 0.4, 0.5], [0, 200, 225, 250]),  # 1.14 -> 1.17
      ([0.3, 0.4, 0.5], [0, 225, 250, 275]),  # 1.17 -> 1.2
      ([0.4, 0.5], [0, 275, 300]),  # 1.2 -> 1.23
      ([0.5, 0.6], [0, 300, 325]),  # 1.23 -> 1.26
      ([0.5, 0.6], [0, 325, 350])  # 1.26 -> 5.0
  ]
  pr_id = bisect.bisect(pressure_ratio_intervals, pressure_ratio)
  soc_id = bisect.bisect(soc_mappings[pr_id][0], state_of_charge)
  return soc_mappings[pr_id][1][soc_id]
