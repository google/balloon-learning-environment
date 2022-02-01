Training and Evaluating a New Agent
===================================

There are two options for training/evaluating a new agent:

#. Create an agent following the
   :py:class:`Agent<balloon_learning_environment.agents.agent.Agent>`
   interface and use our scripts.
#. Use the Balloon Learning Environment gym environment in your own framework.

We recommend following our agent interface for single GPU training. For more
complicated use cases, such as distributed training, it may be easier to
use the gym environment in your own framework.

Using the Agent Interface
#########################

To make a new agent using our framework, you should do the following:

#. Create an agent following the
   :py:class:`Agent<balloon_learning_environment.agents.agent.Agent>`
   interface.
#. Create a training script that uses
   :py:func:`train_lib.run_training_loop<balloon_learning_environment.train_lib.run_training_loop>`.
#. Create an evaluation script that uses
   :py:func:`eval_lib.evaluate_agent<balloon_learning_environment.eval.eval_lib.eval_agent>`.

Creating an Agent
*****************

First, create an agent following the
:py:class:`Agent<balloon_learning_environment.agents.agent.Agent>`
interface. For example:

.. code-block:: python

   from typing import Sequence
   from balloon_learning_environment.agents import agent

   class MyNewAgent(agent.Agent):
       def __init__(self, num_actions: int, observation_shape: Sequence[int]):
           super(MyNewAgent, self).__init__(num_actions, observation_shape)

       ...

Make sure to override all the functions required and recommended by the
:py:class:`Agent<balloon_learning_environment.agents.agent.Agent>` interface.
There are several good examples in the
:doc:`balloon_learning_environment.agents<src/agents>` module.

Create Training Script
**********************

Once you have created your agent, it should be ready to train by calling
:py:func:`train_lib.run_training_loop<balloon_learning_environment.train_lib.run_training_loop>`.
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
:py:class:`Collector<balloon_learning_environment.metrics.collector.Collector>`
to generate statistics during training. We include a set of collectors to

#. print statistics to the console with
   :py:class:`ConsoleCollector<balloon_learning_environment.metrics.console_collector.ConsoleCollector>`
#. save statistics to a pickle file with
   :py:class:`PickleCollector<balloon_learning_environment.metrics.pickle_collector.PickleCollector>`
#. write to Tensorboard event files with
   :py:class:`TensorboardCollector<balloon_learning_environment.metrics.tensorboard_collector.TensorboardCollector>`

You can create your own by following the
:py:class:`Collector<balloon_learning_environment.metrics.collector.Collector>`
signature and passing its constructor to the
:py:class:`CollectorDispatcher<balloon_learning_environment.metrics.collector_dispatcher.CollectorDispatcher>`
.



Create Evaluation Script
************************

If your agent follows the
:py:class:`Agent<balloon_learning_environment.agents.agent.Agent>`
interface, you can also make use of
:py:func:`eval_lib.evaluate_agent<balloon_learning_environment.eval.eval_lib.eval_agent>`.
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
:py:class:`BalloonEnv <balloon_learning_environment.env.balloon_env.BalloonEnv>`.
