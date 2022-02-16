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

"""Example running Acme's distributed QrDQN in JAX on the BLE."""

import functools
from typing import Any, Dict

from absl import app
from absl import flags
from acme import core
from acme import environment_loop
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.jax.layouts import distributed_layout
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
from balloon_learning_environment import acme_utils
import launchpad as lp


FLAGS = flags.FLAGS
flags.DEFINE_string('base_dir', '~/acme',
                    'Directory where to store statistics/images.')
flags.DEFINE_integer('num_actors', 128, 'Number of actors.')
flags.DEFINE_integer('num_episodes', 10000, 'Number of episodes to train for.')
flags.DEFINE_integer('max_episode_length', 960,
                     'Maximum number of steps per episode. Assuming 3 days, '
                     'with each step lasting 3 minutes.')


def default_evaluator(
    environment_factory: types.EnvironmentFactory,
    network_factory: distributed_layout.NetworkFactory,
    policy_factory: distributed_layout.PolicyFactory
) -> distributed_layout.EvaluatorFactory:
  """Returns a default evaluator process."""
  def evaluator(
      random_key: networks_lib.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor: distributed_layout.MakeActorFn,
  ):
    """The evaluation process."""

    # Create environment and evaluator networks
    dummy_seed = 1
    environment = environment_factory(dummy_seed)
    networks = network_factory(specs.make_environment_spec(environment))

    actor = make_actor(
        random_key, policy_factory(networks), variable_source)
    actor._per_episode_update = True  # pylint: disable=protected-access

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = loggers.make_default_logger('evaluator')

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(environment, actor, counter,
                                              logger)
  return evaluator


def get_program(params: Dict[str, Any]) -> lp.Program:
  """Constructs the program."""
  max_episode_length = FLAGS.max_episode_length
  env_factory = functools.partial(
      acme_utils.create_env, max_episode_length=max_episode_length)
  seed = params.pop('seed', 0)

  (rl_agent, config, dqn_network_fn, behavior_policy_fn, eval_policy_fn
   ) = acme_utils.create_dqn(params)

  agent = distributed_layout.DistributedLayout(
      seed=seed,
      environment_factory=lambda seed: env_factory(False),
      network_factory=dqn_network_fn,
      builder=rl_agent,
      policy_network=behavior_policy_fn,
      evaluator_factories=[
          default_evaluator(
              environment_factory=lambda seed: env_factory(True),
              network_factory=dqn_network_fn,
              policy_factory=eval_policy_fn),
      ],
      num_actors=FLAGS.num_actors,
      max_number_of_steps=FLAGS.num_episodes * FLAGS.max_episode_length,
      prefetch_size=config.prefetch_size,
      checkpointing_config=distributed_layout.CheckpointingConfig(
          directory=FLAGS.base_dir,
          add_uid=True,
      ),
  )
  return agent.build()


def main(_):
  program = get_program({
      'seed': 0,
      'num_sgd_steps_per_step': 2,
      'prefetch_size': 0
  })

  # Launch experiment.
  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == '__main__':
  app.run(main)
