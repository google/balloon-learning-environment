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

r"""Main entry point for the Balloon Learning Environment.

"""

import os.path as osp

from absl import app
from absl import flags
from balloon_learning_environment import train_lib
from balloon_learning_environment.env import generative_wind_field
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.rendering import matplotlib_renderer
from balloon_learning_environment.utils import run_helpers
import gym
import matplotlib
import numpy as np


flags.DEFINE_string('agent', 'dqn', 'Type of agent to create.')
flags.DEFINE_string('env_name', 'BalloonLearningEnvironment-v0',
                    'Name of environment to create.')
flags.DEFINE_integer('num_iterations', 200, 'Number of episodes to train for.')
flags.DEFINE_integer('max_episode_length', 960,
                     'Maximum number of steps per episode. Assuming 2 days, '
                     'with each step lasting 3 minutes.')
flags.DEFINE_string('base_dir', None,
                    'Directory where to store statistics/images.')
flags.DEFINE_integer(
    'run_number', 1,
    'When running multiple agents in parallel, this number '
    'differentiates between the runs. It is appended to base_dir.')
flags.DEFINE_string(
    'wind_field', 'generative',
    'The wind field type to use. See the _WIND_FIELDS dict below for options.')
flags.DEFINE_string('agent_gin_file', None,
                    'Gin file for agent configuration.')
flags.DEFINE_multi_string('collectors', ['console'],
                          'Collectors to include in metrics collection.')
flags.DEFINE_multi_string('gin_bindings', [],
                          'Gin bindings to override default values.')
flags.DEFINE_string(
    'renderer', None,
    'The renderer to use. Note that it is fastest to have this set to None.')
flags.DEFINE_integer(
    'render_period', 10,
    'The period to render with. Only has an effect if renderer is not None.')
flags.DEFINE_integer(
    'episodes_per_iteration', 50,
    'The number of episodes to run in one iteration. Checkpointing occurs '
    'at the end of each iteration.')
flags.mark_flag_as_required('base_dir')
FLAGS = flags.FLAGS


_WIND_FIELDS = {
    'generative': generative_wind_field.GenerativeWindField,
    'simple': wind_field.SimpleStaticWindField,
}

_RENDERERS = {
    'matplotlib': matplotlib_renderer.MatplotlibRenderer,
}


def main(_) -> None:
  # Prepare metric collector gin files and constructors.
  collector_constructors = train_lib.get_collector_data(FLAGS.collectors)
  run_helpers.bind_gin_variables(FLAGS.agent,
                                 FLAGS.agent_gin_file,
                                 FLAGS.gin_bindings)

  renderer = None
  if FLAGS.renderer is not None:
    renderer = _RENDERERS[FLAGS.renderer]()

  wf = _WIND_FIELDS[FLAGS.wind_field]
  env = gym.make(FLAGS.env_name,
                 wind_field_factory=wf,
                 renderer=renderer)

  agent = run_helpers.create_agent(
      FLAGS.agent,
      env.action_space.n,
      observation_shape=env.observation_space.shape)

  base_dir = osp.join(FLAGS.base_dir, FLAGS.agent, str(FLAGS.run_number))
  train_lib.run_training_loop(
      base_dir,
      env,
      agent,
      FLAGS.num_iterations,
      FLAGS.max_episode_length,
      collector_constructors,
      render_period=FLAGS.render_period,
      episodes_per_iteration=FLAGS.episodes_per_iteration)

  if FLAGS.base_dir is not None:
    image_save_path = osp.join(FLAGS.base_dir, 'balloon_path.png')
    img = env.render(mode='rgb_array')
    if isinstance(img, np.ndarray):
      matplotlib.image.imsave(image_save_path, img)


if __name__ == '__main__':
  app.run(main)
