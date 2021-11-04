# Balloon Learning Environment

The Balloon Learning Environment (BLE) is a simulator for training RL agents
to control stratospheric balloons. It is a followup to the Nature paper
["Autonomous navigation of stratospheric balloons using reinforcement learning"](https://www.nature.com/articles/s41586-020-2939-8).

## Installation

For now, the BLE can be used by cloning the source:

```
git clone https://github.com/google/balloon-learning-environment
```

```
cd balloon-learning-environment
```

We recommend using a virtual environment:

```
python -m venv .venv && source .venv/bin/activate
```

Make sure pip is the latest version:

```
pip install --upgrade pip
```

Install all the prerequisites:

```
pip install -r requirements.txt
```

The BLE internally uses a neural network as part of the environment, so we
recommend installing jax with GPU support.
See the [jax codebase](https://github.com/google/jax#pip-installation-gpu-cuda)
for instructions.

## Evaluating an Agent

See the [evaluation readme](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/README.md) for instructions on evaluating an agent.

