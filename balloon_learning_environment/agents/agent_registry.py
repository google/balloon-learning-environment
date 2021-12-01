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

"""The registry of agents.

This is where you add new agents; we provide some examples to get you started.

When writing a new agent, follow the API specified by the base class
`agent.Agent` and implement the abstract methods.
The provided agents are:
  RandomAgent: Ignores all observations and picks actions randomly.
  MLPAgent: Uses a simple multi-layer perceptron (MLP) to learn the mapping of
    states to Q-values. The number of layers and hidden units in the MLP is
    configurable.
"""

from typing import Callable, Optional

from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents import dqn_agent
from balloon_learning_environment.agents import mlp_agent
from balloon_learning_environment.agents import perciatelli44
from balloon_learning_environment.agents import quantile_agent
from balloon_learning_environment.agents import random_walk_agent
from balloon_learning_environment.agents import station_seeker_agent

BASE_DIR = 'balloon_learning_environment/agents/configs'
REGISTRY = {
    'random': (agent.RandomAgent, None),
    'mlp': (mlp_agent.MLPAgent, f'{BASE_DIR}/mlp.gin'),
    'dqn': (dqn_agent.DQNAgent, f'{BASE_DIR}/dqn.gin'),
    'perciatelli44': (perciatelli44.Perciatelli44, None),
    'quantile': (quantile_agent.QuantileAgent, f'{BASE_DIR}/quantile.gin'),
    'finetune_perciatelli': (quantile_agent.QuantileAgent,
                             f'{BASE_DIR}/finetune_perciatelli.gin'),
    'station_seeker': (station_seeker_agent.StationSeekerAgent, None),
    'random_walk': (random_walk_agent.RandomWalkAgent, None),
}


def agent_constructor(name: str) -> Callable[..., agent.Agent]:
  if name not in REGISTRY:
    raise ValueError(f'Agent {name} not recognized')
  return REGISTRY[name][0]


def get_default_gin_config(name: str) -> Optional[str]:
  if name not in REGISTRY:
    raise ValueError(f'Agent {name} not recognized')
  return REGISTRY[name][1]
