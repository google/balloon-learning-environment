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

"""Functions used by the main train binary."""

import os.path as osp
from typing import Iterable, List, Optional, Sequence, Tuple

from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.metrics import collector_dispatcher
from balloon_learning_environment.metrics import statistics_instance


def get_collector_data(
    collectors: Optional[Iterable[str]] = None
) -> Tuple[List[str], List[collector_dispatcher.CollectorConstructorType]]:
  """Returns a list of gin files and constructors for each passed collector."""
  gin_files = []
  collector_constructors = []
  for c in collectors:
    if c not in collector_dispatcher.AVAILABLE_COLLECTORS:
      continue
    gin_files.append(
        f'{collector_dispatcher.BASE_CONFIG_PATH}/{c}_collector.gin')
    collector_constructors.append(
        collector_dispatcher.AVAILABLE_COLLECTORS[c])
  return gin_files, collector_constructors


def run_training_loop(
    base_dir: str,
    env: balloon_env.BalloonEnv,
    agent: base_agent.Agent,
    num_episodes: int,
    max_episode_length: int,
    collector_constructors: Sequence[
        collector_dispatcher.CollectorConstructorType],
    *,
    render_period: int = 10) -> None:
  """Run a training loop for a specified number of steps."""
  checkpoint_dir = osp.join(base_dir, 'checkpoints')
  # Possibly reload the latest checkpoint, and start from the next episode
  # number.
  start_episode = agent.reload_latest_checkpoint(checkpoint_dir) + 1
  dispatcher = collector_dispatcher.CollectorDispatcher(
      base_dir, env.action_space.n, collector_constructors, start_episode)
  # Maybe pass on a summary writer to the environment.
  env.set_summary_writer(dispatcher.get_summary_writer())
  # Maybe pass on a sumary writer to the agent.
  agent.set_summary_writer(dispatcher.get_summary_writer())

  agent.set_mode(base_agent.AgentMode.TRAIN)

  dispatcher.pre_training()
  for episode in range(start_episode, num_episodes):
    dispatcher.begin_episode()
    obs = env.reset()
    # Request first action from agent.
    a = agent.begin_episode(obs)
    terminal = False
    final_episode_step = max_episode_length
    r = 0.0

    for i in range(max_episode_length):
      # Pass action to environment.
      obs, r, terminal, _ = env.step(a)

      if i % render_period == 0:
        env.render()  # No-op if renderer is None.

      # Record the current transition.
      dispatcher.step(statistics_instance.StatisticsInstance(
          step=i,
          action=a,
          reward=r,
          terminal=terminal))

      if terminal:
        final_episode_step = i + 1
        break

      # Pass observation to agent, request new action.
      a = agent.step(r, obs)

    # The environment has no timeout, so terminal really is a terminal state.
    agent.end_episode(r, terminal)
    # Possibly checkpoint the agent.
    agent.save_checkpoint(checkpoint_dir, episode)
    # TODO(joshgreaves): Fix dispatcher logging the same data twice on terminal.
    dispatcher.end_episode(statistics_instance.StatisticsInstance(
        step=final_episode_step,
        action=a,
        reward=r,
        terminal=terminal))

  dispatcher.end_training()
