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

"""Functions used by the main train binary."""

import os.path as osp
from typing import Iterable, List, Optional, Sequence

from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.metrics import collector_dispatcher
from balloon_learning_environment.metrics import statistics_instance


def get_collector_data(
    collectors: Optional[Iterable[str]] = None
) -> List[collector_dispatcher.CollectorConstructorType]:
  """Returns a list of gin files and constructors for each passed collector."""
  collector_constructors = []
  for c in collectors:
    if c not in collector_dispatcher.AVAILABLE_COLLECTORS:
      continue
    collector_constructors.append(
        collector_dispatcher.AVAILABLE_COLLECTORS[c])
  return collector_constructors


def _run_one_episode(env: balloon_env.BalloonEnv,
                     agent: base_agent.Agent,
                     dispatcher: collector_dispatcher.CollectorDispatcher,
                     max_episode_length: int,
                     render_period: int) -> None:
  """Runs an agent in an environment for one episode."""
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
    dispatcher.step(
        statistics_instance.StatisticsInstance(
            step=i, action=a, reward=r, terminal=terminal))

    if terminal:
      final_episode_step = i + 1
      break

    # Pass observation to agent, request new action.
    a = agent.step(r, obs)

  # The environment has no timeout, so terminal really is a terminal state.
  agent.end_episode(r, terminal)

  dispatcher.end_episode(
      statistics_instance.StatisticsInstance(
          step=final_episode_step, action=a, reward=r, terminal=terminal))


def run_training_loop(base_dir: str,
                      env: balloon_env.BalloonEnv,
                      agent: base_agent.Agent,
                      num_iterations: int,
                      max_episode_length: int,
                      collector_constructors: Sequence[
                          collector_dispatcher.CollectorConstructorType],
                      *,
                      render_period: int = 10,
                      episodes_per_iteration: int = 50) -> None:
  """Runs a training loop for a specified number of steps.

  Args:
    base_dir: The directory to use as the experiment root. This is where
      checkpoints and collector outputs will be written.
    env: The environment to train on.
    agent: The agent to train.
    num_iterations: The number of iterations to train for.
    max_episode_length: The number of episodes at which to end an episode.
    collector_constructors: A sequence of collector constructors for
      collecting and reporting training statistics.
    render_period: The period with which to render the environment. This only
      has an effect if the environments renderer is not None.
    episodes_per_iteration: The number of episodes to run per iteration.
  """
  checkpoint_dir = osp.join(base_dir, 'checkpoints')
  # Possibly reload the latest checkpoint, and start from the next episode
  # number.
  start_iteration = max(agent.reload_latest_checkpoint(checkpoint_dir), 0)
  dispatcher = collector_dispatcher.CollectorDispatcher(
      base_dir,
      env.action_space.n,
      collector_constructors,
      start_iteration * episodes_per_iteration)
  # Maybe pass on a sumary writer to the agent.
  agent.set_summary_writer(dispatcher.get_summary_writer())

  agent.set_mode(base_agent.AgentMode.TRAIN)

  dispatcher.pre_training()
  for iteration in range(start_iteration, num_iterations):
    for _ in range(episodes_per_iteration):
      _run_one_episode(env,
                       agent,
                       dispatcher,
                       max_episode_length,
                       render_period)

    agent.save_checkpoint(checkpoint_dir, iteration)

  dispatcher.end_training()
