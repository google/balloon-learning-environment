Training and Evaluating a New Agent
===================================

There are two options for training/evaluating a new agent:

#. Create an agent following the
   `Agent <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/agents/agent.py>`_
   interface and use our scripts.
#. Use the Balloon Learning Environment gym environment in your own framework.

We recommend following our agent interface for single GPU training. For more
complicated use cases, such as distributed training, it may be easier to
use the gym environment in your own framework.

Using the Agent Interface
#########################

To make a new agent using our framework, you should do the following:

#. Create an agent following the
   `Agent <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/agents/agent.py>`_
   interface.
#. Create a training script that uses
   `train_lib.run_training_loop <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/train_lib.py>`_.
#. Create an evaluation script that uses
   `eval_lib.eval_agent <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/eval_lib.py>`_.

Creating an Agent
*****************

First, create an agent following the
`Agent <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/agents/agent.py>`_
interface. For example:

.. code-block:: python

   from typing import Sequence
   from balloon_learning_environment.agents import agent

   class MyNewAgent(agent.Agent):
       def __init__(self, num_actions: int, observation_shape: Sequence[int]):
           super(MyNewAgent, self).__init__(num_actions, observation_shape)

       ...

Make sure to override all the functions required and recommended by the
`Agent <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/agents/agent.py>`_
There are several good examples in the
`agents <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/agents>`_
module.


Alternatively, you can use one of the agents provided with the BLE:

.. code-block:: python

   from balloon_learning_environment import train_lib
   from balloon_learning_environment.env import balloon_env
   from balloon_learning_environment.utils import run_helpers

   agent_name = 'quantile'
   env = gym.make('BalloonLearningEnvironment-v0')
   run_helpers.bind_gin_variables(agent_name)
   agent = run_helpers.create_agent(agent_name,
                                    env.action_space.n,
                                    env.observation_space.shape)


Create Training Script
**********************

Once you have created your agent, it should be ready to train by calling
`train_lib.run_training_loop <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/train_lib.py>`_.
You'll need to create a launch script that sets up the environment and agent
and calls this function.

For an example, take a look at our
`train.py <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/train.py>`_
which we use to train the benchmark agents. A slimmed down version would
look something like this:

.. code-block:: python

   from balloon_learning_environment import train_lib
   from balloon_learning_environment.env import balloon_env  # Registers the environment.
   import gym

   env = gym.make('BalloonLearningEnvironment-v0')
   agent = YourAgent(env.action_space.n, env.observation_space.shape)

   train_lib.run_training_loop(
       '/tmp/ble/train/my_experiment',  # The experiment root path.
       env,
       agent,
       num_iterations=2000,
       max_episode_length=960,  # 960 steps is 2 days, the default amount.
       collector_constructors=[])  # Specify some collectors to log training stats.


You can optionally add
`Collectors <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/metrics/collector.py>`_
to generate statistics during training. We include a set of collectors to

#. print statistics to the console with
   `ConsoleCollector <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/metrics/console_collector.py>`_.
#. save statistics to a pickle file with
   `PickleCollector <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/metrics/pickle_collector.py>`_.
#. write to Tensorboard event files with
   `TensorboardCollector <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/metrics/tensorboard_collector.py>`_.

You can create your own by following the
`Collector <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/metrics/collector.py>`_
interface and passing its constructor to the
`CollectorDispatcher <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/metrics/collector_dispatcher.py>`_.



Create Evaluation Script
************************

If your agent follows the
`Agent <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/agents/agent.py>`_
interface, you can also make use of
`eval_lib.eval_agent <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/eval_lib.py>`_.
Once again, you'll need to create a launch script that sets up the environment
and agent, and then calls the function.

For an example, take a look at our
`eval.py <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/eval.py>`_
which we use to evaluate the benchmark agents. A slimmed down version would
look something like this:

.. code-block:: python

   from balloon_learning_environment.env import balloon_env  # Registers the environment.
   from balloon_learning_environment.eval import eval_lib
   import gym

   env = gym.make('BalloonLearningEnvironment-v0')
   agent = YourAgent(env.action_space.n, env.observation_space.shape)

   eval_results = eval_agent(
       agent,
       env,
       eval_suite=suites.get_eval_suite('small_eval'))

  do_something_with_eval_results(eval_results)  # Write to disk, for example.

'small_eval' uses 100 seeds, which may take around 3 GPU hours, depending on
the GPU. 'small_eval' is great for determining the progress of an agent.
Once you are satisfied with an agent, we recommend reporting 'big_eval'
results where feasible. 'big_eval' uses 10,000 seeds, and takes around 300
GPU hours. This work can be parallelized and spread out across multiple shards,
as we demonstrate in
`eval.py <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/eval.py>`_.


Using a Different Framework
###########################

If you choose to use a different framework for training an agent, simply create
an environment and interact with it in the way that makes sense for your
framework or experiment.

.. code-block:: python

   from balloon_learning_environment.env import balloon_env  # Registers the environment.
   import gym

   env = gym.make('BalloonLearningEnvironment-v0')
   # Do what you want with the environment now it has been created.

The environment follows the standard gym interface. The type of the returned
environment object is
`BalloonEnv <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/env/balloon_env.py>`_.
