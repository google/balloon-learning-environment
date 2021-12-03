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

r"""Entry point for evaluating agents on the Balloon Learning Environment.

"""

import json
import os
from typing import Sequence

from absl import app
from absl import flags
from balloon_learning_environment.env import balloon_env  # pylint: disable=unused-import
from balloon_learning_environment.env import generative_wind_field
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.rendering import matplotlib_renderer
from balloon_learning_environment.eval import eval_lib
from balloon_learning_environment.eval import suites
from balloon_learning_environment.utils import run_helpers
import gym


flags.DEFINE_string('agent', 'dqn', 'The name of the agent to create.')
flags.DEFINE_enum('suite', 'big_eval',
                  suites.available_suites(),
                  'The evaluation suite to run.')
flags.DEFINE_string(
    'wind_field', 'generative',
    'The windfield type to use. See the _WIND_FIELDS dict below for options.')
flags.DEFINE_string('agent_gin_file', None, 'Gin file for agent configuration.')
flags.DEFINE_multi_string('gin_bindings', [],
                          'Gin bindings to override default values.')
flags.DEFINE_string('output_dir', '/tmp/ble/eval',
                    'The directory to output the json summary.')
flags.DEFINE_integer(
    'num_shards',
    1,
    'The number of per-agent shards to split the eval job into.',
    lower_bound=1)
flags.DEFINE_integer('shard_idx', 0, 'The index of the shard.', lower_bound=0)
flags.DEFINE_boolean('pretty_json', False,
                     'If true, it will write json files with an indent of 2.')
flags.DEFINE_string('checkpoint_dir', None,
                    'The directory to load checkpoints from.')
flags.DEFINE_integer('checkpoint_idx', None,
                     'The checkpoint iteration number to load.')
flags.DEFINE_string(
    'name_override', None,
    'If supplied, this will be the name used for the json output file.')
flags.DEFINE_string(
    'renderer', None,
    'The renderer to use. Note that it is fastest to have this set to None.')
flags.DEFINE_integer(
    'render_period', 10,
    'The period to render with. Only has an effect if renderer is not None.')
FLAGS = flags.FLAGS


_WIND_FIELDS = {
    'generative': generative_wind_field.GenerativeWindField,
    'simple': wind_field.SimpleStaticWindField,
}

_RENDERERS = {
    'matplotlib': matplotlib_renderer.MatplotlibRenderer,
}


def write_result(result: Sequence[eval_lib.EvaluationResult]) -> None:
  """Writes an evaluation result as a json file."""
  if FLAGS.name_override:
    file_name = FLAGS.name_override
  elif FLAGS.checkpoint_idx is not None:
    file_name = f'{FLAGS.agent}_{FLAGS.checkpoint_idx}'
  else:
    file_name = FLAGS.agent

  if FLAGS.num_shards > 1:
    file_name = f'{file_name}_{FLAGS.shard_idx}'
  file_name = f'{file_name}.json'

  dir_path = os.path.join(FLAGS.output_dir, FLAGS.suite)
  file_path = os.path.join(dir_path, file_name)

  indent = 2 if FLAGS.pretty_json else None


  os.makedirs(dir_path, exist_ok=True)
  with open(file_path, 'w') as f:
    json.dump(result, f, cls=eval_lib.EvalResultEncoder, indent=indent)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_helpers.bind_gin_variables(FLAGS.agent,
                                 FLAGS.agent_gin_file,
                                 FLAGS.gin_bindings)

  renderer = None
  if FLAGS.renderer is not None:
    renderer = _RENDERERS[FLAGS.renderer]()

  wf = _WIND_FIELDS[FLAGS.wind_field]
  env = gym.make('BalloonLearningEnvironment-v0',
                 wind_field_factory=wf,
                 renderer=renderer)

  agent = run_helpers.create_agent(
      FLAGS.agent,
      env.action_space.n,
      observation_shape=env.observation_space.shape)
  if FLAGS.checkpoint_dir is not None and FLAGS.checkpoint_idx is not None:
    agent.load_checkpoint(FLAGS.checkpoint_dir, FLAGS.checkpoint_idx)

  eval_suite = suites.get_eval_suite(FLAGS.suite)

  if FLAGS.num_shards > 1:
    start = int(len(eval_suite.seeds) * FLAGS.shard_idx / FLAGS.num_shards)
    end = int(len(eval_suite.seeds) * (FLAGS.shard_idx + 1) / FLAGS.num_shards)
    eval_suite.seeds = eval_suite.seeds[start:end]

  eval_result = eval_lib.eval_agent(agent, env, eval_suite,
                                    render_period=FLAGS.render_period)
  write_result(eval_result)


if __name__ == '__main__':
  app.run(main)
