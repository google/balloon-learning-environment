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

"""Tests for eval_lib."""

import datetime as dt
import json

from absl.testing import absltest
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.eval import eval_lib
from balloon_learning_environment.eval import suites
from balloon_learning_environment.utils import run_helpers
from balloon_learning_environment.utils import test_helpers
from balloon_learning_environment.utils import units
import jax
import numpy as np


class EvalLibTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_helpers.bind_environment_gin_parameters(seed=0)
    self.env = balloon_env.BalloonEnv()
    self.env.arena._step_duration = dt.timedelta(seconds=10)
    self.agent = run_helpers.create_agent('random', self.env.action_space.n,
                                          self.env.observation_space.shape)
    self.eval_suite = suites.EvaluationSuite(
        seeds=range(2), max_episode_length=3)

    np.random.seed(0)  # Required for random agent.

  def test_eval_agent_returns_same_number_of_results_as_seeds(self):
    result = eval_lib.eval_agent(self.agent, self.env, self.eval_suite)

    self.assertLen(result, len(self.eval_suite.seeds))

  def test_eval_agent_is_deterministic_with_deterministic_agent(self):
    result1 = eval_lib.eval_agent(self.agent, self.env, self.eval_suite)
    np.random.seed(0)  # Random agent must be re-seeded to be deterministic.
    result2 = eval_lib.eval_agent(self.agent, self.env, self.eval_suite)

    # Cumulative rewards vary greatly based on position, so they are
    # enough to distinguish between the same and alternate routes.
    cumulative_rewards1 = [x.cumulative_reward for x in result1]
    cumulative_rewards2 = [x.cumulative_reward for x in result2]
    self.assertEqual(cumulative_rewards1, cumulative_rewards2)

  def test_eval_agent_with_different_seeds_gives_different_result(self):
    result = eval_lib.eval_agent(self.agent, self.env, self.eval_suite)

    self.assertNotEqual(result[0], result[1])

  def test_eval_agent_returns_twr_within_correct_range(self):
    # The correct range for TWR is [0, 1].
    result = eval_lib.eval_agent(self.agent, self.env, self.eval_suite)
    twrs_are_in_range = all(0.0 <= x.time_within_radius <= 1.0 for x in result)

    self.assertTrue(twrs_are_in_range)

  def test_eval_result_encoder_successfully_encodes_object(self):
    seed = 10
    cumulative_reward = 20.0
    time_within_radius = 5.5
    out_of_power = False
    envelope_burst = False
    zeropressure = False
    final_timestep = 1
    power_percent = 0.95
    atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(0))
    flight_path = [
        eval_lib.SimpleBalloonState.from_balloon_state(
            test_helpers.create_balloon(
                x=units.Distance(m=1.0),
                y=units.Distance(m=2.0),
                pressure=3.0,
                time_elapsed=dt.timedelta(days=5),
                power_percent=power_percent,
                atmosphere=atmosphere).state)
    ]
    eval_result = eval_lib.EvaluationResult(
        seed=seed,
        cumulative_reward=cumulative_reward,
        time_within_radius=time_within_radius,
        out_of_power=out_of_power,
        envelope_burst=envelope_burst,
        zeropressure=zeropressure,
        final_timestep=final_timestep,
        flight_path=flight_path)

    eval_string = json.dumps(eval_result, cls=eval_lib.EvalResultEncoder)
    eval_dict = json.loads(eval_string)

    self.assertEqual(eval_dict['seed'], seed)
    self.assertEqual(eval_dict['cumulative_reward'], cumulative_reward)
    self.assertEqual(eval_dict['time_within_radius'], time_within_radius)
    self.assertEqual(eval_dict['out_of_power'], out_of_power)
    self.assertEqual(eval_dict['envelope_burst'], envelope_burst)
    self.assertEqual(eval_dict['zeropressure'], zeropressure)
    self.assertEqual(eval_dict['final_timestep'], final_timestep)

    flight_path_point = eval_dict['flight_path'][0]
    self.assertAlmostEqual(flight_path_point['x'], flight_path[0].x.kilometers)
    self.assertAlmostEqual(flight_path_point['y'], flight_path[0].y.kilometers)
    self.assertAlmostEqual(flight_path_point['pressure'],
                           flight_path[0].pressure)
    self.assertAlmostEqual(flight_path_point['elapsed_seconds'],
                           flight_path[0].time_elapsed.total_seconds())
    self.assertAlmostEqual(flight_path_point['power'], power_percent)


if __name__ == '__main__':
  absltest.main()
