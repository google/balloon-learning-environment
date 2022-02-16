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

"""Balloon Learning Environment gym utilities."""
import contextlib


def register_env() -> None:
  """Register the Gym environment."""
  # We need to import Gym's registration module inline or else we'll
  # get a circular dependency that will result in an error when importing gym
  from gym.envs import registration  # pylint: disable=g-import-not-at-top

  env_id = 'BalloonLearningEnvironment-v0'
  env_entry_point = 'balloon_learning_environment.env.balloon_env:BalloonEnv'
  # We guard registration by checking if our env is already registered
  # This is necesarry because the plugin system will load our module
  # which also calls this function. If multiple `register()` calls are
  # made this will result in a warning to the user.
  registered = env_id in registration.registry.env_specs

  if not registered:
    with contextlib.ExitStack() as stack:
      # This is a workaround for Gym 0.21 which didn't support
      # registering into the root namespace with the plugin system.
      if hasattr(registration, 'namespace'):
        stack.enter_context(registration.namespace(None))
      registration.register(id=env_id, entry_point=env_entry_point)
