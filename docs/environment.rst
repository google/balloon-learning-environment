Using and Configuring the Environment
=====================================

Basic Usage
###########

The main entrypoint to the BLE for most users is the gym environment. To use
the environment, import the balloon environment and use gym to create it:

.. code-block:: python

   import balloon_learning_environment.env.balloon_env  # Registers the environment.
   import gym

   env = gym.make('BalloonLearningEnvironment-v0')


This will give you a new
`BalloonEnv <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/env/balloon_env.py>`_
object that follows the gym environment interface. Before we run the
environment we can inspect the observation and action spaces:

.. code-block:: python

   >>> print(env.observation_space)
   Box([0. 0. 0. ... 0. 0. 0.], [1. 1. 1. ... 1. 1. 1.], (1099,), float32)
   >>> print(env.action_space)
   Discrete(3)


Here we can see that the observation space is a 1099 dimensional array,
and the action space has 3 discrete actions. We can use the environment
as follows:

.. code-block:: python

   env.seed(0)
   observation_0 = env.reset()
   observation_1, reward, is_terminal, info = env.step(0)


In this snippet, we seeded the environment to give it a deterministic
initialization. This is useful for replicating results (for example, in
evaluation), but most of the time you'll want to skip this line to have
a random initialization. After seeding the environment we reset it and
stepped once with action 0.

We expect the observations to have the shape specified by observation_space:

.. code-block:: python

   >>> print(type(observation_0), observation_0.shape)
   <class 'numpy.ndarray'> (1099,)


The reward, is_terminal, and info objects are as follows:

.. code-block:: python

   >>> print(reward, is_terminal, info, sep='\n')
   0.26891435801077535
   False
   {'out_of_power': False, 'envelope_burst': False, 'zeropressure': False, 'time_elapsed': datetime.timedelta(seconds=180)}


These should be enough to start training an RL agent.

Configuring the Environment
###########################

The environment may be configured to give custom behavior. To see all
options for configuring the environment, see the
`BalloonEnv <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/env/balloon_env.py>`_
constructor. Here, we highlight important options.

First, the
`FeatureConstructor <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/env/features.py>`_
class may be swapped out. The feature constructor receives observations
from the simulator at each step, and returns features when required. This
setup allows a feature constructor to maintain its own state, and use the
simulator history to create a feature vector. The default feature constructor
is the
`PerciatelliFeatureConstructor <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/env/features.py>`_.

The reward function can also be swapped out. The default reward function,
`perciatelli_reward_function <https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/env/balloon_env.py>`_
gives a reward of 1.0 as long as the agent is in the stationkeeping readius.
The reward decays exponentially outside of this radius.

.. image:: imgs/reward_function.png

