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

"""Common constants used throughout the codebase."""

import datetime as dt


# --- Physics Constants ---

GRAVITY: float = 9.80665  # [m/s^2]
NUM_SECONDS_PER_HOUR = 3_600
NUM_SECONDS_PER_DAY: int = 86_400
UNIVERSAL_GAS_CONSTANT: float = 8.3144621  # [J/(mol.K)]
DRY_AIR_MOLAR_MASS: float = 0.028964922481160  # Dry Air. [kg/mol]
HE_MOLAR_MASS: float = 0.004002602  # Helium.  [kg/mol]
DRY_AIR_SPECIFIC_GAS_CONSTANT: float = (
    UNIVERSAL_GAS_CONSTANT / DRY_AIR_MOLAR_MASS)  # [J/(kg.K)]


# --- RL constants ---
# Amount of time that elapses between agent steps.
AGENT_TIME_STEP: dt.timedelta = dt.timedelta(minutes=3)
# Pressure limits for the Perciatelli features.
PERCIATELLI_PRESSURE_RANGE_MIN: int = 5000
PERCIATELLI_PRESSURE_RANGE_MAX: int = 14000

