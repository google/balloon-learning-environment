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

"""Model classes for simulator state and simulator observations."""

import dataclasses

from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import standard_atmosphere


@dataclasses.dataclass
class SimulatorState(object):
  """Specifies the full state of the simulator.

  Since it specifies the full state of the simulator, it should be
  possible to use this for checkpointing and restoring the simulator.
  """
  balloon_state: balloon.BalloonState
  wind_field: wind_field.WindField
  atmosphere: standard_atmosphere.Atmosphere


@dataclasses.dataclass
class SimulatorObservation(object):
  """Specifies an observation from the simulator.

  This differs from SimulatorState in that the observations are not
  ground truth state, and are instead noisy observations from the
  environment.
  """
  balloon_observation: balloon.BalloonState
  wind_at_balloon: wind_field.WindVector
