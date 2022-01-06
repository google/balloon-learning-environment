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

"""Helper functions for running agents in train/eval."""

import os
from typing import Optional, Sequence

import balloon_learning_environment
from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.agents import agent_registry
import gin


def get_agent_gin_file(agent_name: str,
                       gin_file: Optional[str]) -> Optional[str]:
  """Gets a gin file for a specified agent.

  If gin_file is specified, that is the gin_file that will be used.
  However, if no gin file is specified, it will use the default gin file
  for that agent. If the agent has no default gin file, this may return None.

  Args:
    agent_name: The name of the agent to retrieve the gin file for.
    gin_file: An optional gin file to override the agent's default gin file.

  Returns:
    A path to a gin file, or None.
  """
  return (agent_registry.get_default_gin_config(agent_name)
          if gin_file is None else gin_file)


def create_agent(agent_name: str, num_actions: int,
                 observation_shape: Sequence[int]) -> base_agent.Agent:
  return agent_registry.agent_constructor(agent_name)(
      num_actions, observation_shape=observation_shape)


def bind_gin_variables(
    agent: str,
    agent_gin_file: Optional[str] = None,
    gin_bindings: Sequence[str] = (),
    additional_gin_files: Sequence[str] = ()
) -> None:
  """A helper function for binding gin variables for an experiment.

  Args:
    agent: The agent being used in the experiment.
    agent_gin_file: An optional path to a gin file to override the agent's
      default gin file.
    gin_bindings: An optional list of gin bindings passed in on the command
      line.
    additional_gin_files: Any other additional paths to gin files that should be
      parsed and bound.
  """
  gin_files = []

  # The gin file paths start with balloon_learning_environment,
  # so we need to add the parent directory to the search path.
  ble_root = os.path.dirname(balloon_learning_environment.__file__)
  ble_parent_dir = os.path.dirname(ble_root)
  gin.add_config_file_search_path(ble_parent_dir)

  agent_gin_file = get_agent_gin_file(agent, agent_gin_file)
  if agent_gin_file is not None:
    gin_files.append(agent_gin_file)

  gin_files.extend(additional_gin_files)
  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings, skip_unknown=False)
