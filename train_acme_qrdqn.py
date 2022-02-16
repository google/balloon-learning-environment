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

r"""Entry point for Acme QrDQN training on the BLE.
"""


from absl import app
from absl import flags
import acme
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
from balloon_learning_environment import acme_utils
import jax

flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to train for.')
flags.DEFINE_integer('max_episode_length', 960,
                     'Maximum number of steps per episode. Assuming 2 days, '
                     'with each step lasting 3 minutes.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

FLAGS = flags.FLAGS


def main(_) -> None:

  env = acme_utils.create_env(False, FLAGS.max_episode_length)
  eval_env = acme_utils.create_env(True, FLAGS.max_episode_length)
  env_spec = acme.make_environment_spec(env)

  (rl_agent, config, dqn_network_fn, behavior_policy_fn, eval_policy_fn
   ) = acme_utils.create_dqn({})
  dqn_network = dqn_network_fn(env_spec)
  config.samples_per_insert_tolerance_rate = float('inf')
  min_replay_size = config.min_replay_size
  config.min_replay_size = 1

  counter = counting.Counter(time_delta=0.)

  agent = local_layout.LocalLayout(
      seed=FLAGS.seed,
      environment_spec=env_spec,
      builder=rl_agent,
      networks=dqn_network,
      policy_network=behavior_policy_fn(dqn_network),
      batch_size=config.batch_size,
      min_replay_size=min_replay_size,
      samples_per_insert=config.samples_per_insert,
      prefetch_size=4,
      device_prefetch=True,
      counter=counting.Counter(counter, 'learner'))

  eval_actor = rl_agent.make_actor(jax.random.PRNGKey(0),
                                   policy_network=eval_policy_fn(dqn_network),
                                   variable_source=agent)

  actor_logger = loggers.make_default_logger('actor')
  evaluator_logger = loggers.make_default_logger('evaluator')

  loop = acme.EnvironmentLoop(
      env,
      agent,
      logger=actor_logger,
      counter=counting.Counter(counter, 'actor', time_delta=0.))
  eval_loop = acme.EnvironmentLoop(
      eval_env,
      eval_actor,
      logger=evaluator_logger,
      counter=counting.Counter(counter, 'evaluator', time_delta=0.))
  for _ in range(FLAGS.num_episodes):
    loop.run(1)
    eval_loop.run(1)


if __name__ == '__main__':
  app.run(main)
