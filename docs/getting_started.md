# Getting Started

## Installation

To get started with the Balloon Learning Environment, install the package with
pip. First, ensure your pip version is up to date:

```
pip install --upgrade pip
```

and then install the balloon_learning_environment package:

```
pip install balloon_learning_envrionment
```

## Ensure the BLE is Using Your GPU/TPU

The BLE contains a VAE for generating winds, which you will probably want
to run on your accelerator. See the jax documentation for installing with
[GPU](https://github.com/google/jax#pip-installation-gpu-cuda) or
[TPU](https://github.com/google/jax#pip-installation-google-cloud-tpu).

As a sanity check, you can open interactive python and run:

```
from balloon_learning_environment.env import balloon_env
env = balloon_env.BalloonEnv()
```

If you are not running with GPU/TPU, you should see a log like:

```
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
```

If you don't see this log, you should be good to go!

## Training Baseline Agents

Once the Balloon Learning Environment has been installed you can start
training the baseline agents. The BLE has a training script for the baseline
agents that you can call with

```
python -m balloon_learning_environment.train \
  --base_dir=/tmp/ble/train \
  --agent=finetune_perciatelli \
  --renderer=matplotlib
```

Using the matplotlib renderer allows you to see in real time how the agent
is performing. However, the renderer also has a significant overhead.
We recommend using `--renderer=None`, which is the default, for speed.

Other agents you could train are:
*  `dqn`: A simple DQN agent. This is not a strong baseline, and is mostly
   to demonstrate that solving the BLE is not a simple task!
*  `quantile`: A Quantile-based agent in JAX that uses the same
   architecture as Perciatelli44, but starts training from a fresh
   initialization.
*  `finetune_perciatelli`: The same as `quantile`, but reloads the
   `perciatelli44` weights, and is a great way to warm-start agent training.

For more options for using the train script, see the flags at the top of the
[train.py](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/train.py)
file.

## Evaluating Baseline Agents

The BLE also comes with an evaluation suite. This lets us run our agent on
a large number of environment seeds and aggregate the results. To run an
evaluation suite on a benchmark agent, use the following example command:

```
python -m balloon_learning_environment.eval.eval \
  --output_dir=/tmp/ble/eval \
  --agent=random \
  --suite=micro_eval
  --renderer=matplotlib
```

This will evaluate the random agent on 1 seed and write the result to
`/tmp/ble/eval` as a json file. This file can be loaded in the
[summarize_eval](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/colab/summarize_eval.ipynb)
notebook to summarize statistics about the flight.

Other agents to evaluate (including agents mentioned above) are:
*  `perciatelli44`: A state-of-the-art learned agent reported in
   ["Autonomous navigation of stratospheric balloons using reinforcement learning"](https://www.nature.com/articles/s41586-020-2939-8).
*  `station_seeker`: A rule-based agent that achieves good performance, also
   reported in
   ["Autonomous navigation of stratospheric balloons using reinforcement learning"](https://www.nature.com/articles/s41586-020-2939-8).

You can also try evaluation on other suites:
*  `big_eval`: This suite containes 10,000 seeds and gives a good signal of
   how well an agent station-keeps. However, this suite may take up to
   300 hours on a single GPU!
*  `small_eval`: This is a very useful evaluation suite to run. It contains
   100 seeds and gives a rough view into how well an agent performs. On abui
   single GPU, it may take around 3 hours.

For more available suites, see [suites.py](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/suites.py).
