# Balloon Learning Environment

The Balloon Learning Environment (BLE) is a simulator for training RL agents
to control stratospheric balloons. It is a followup to the Nature paper
["Autonomous navigation of stratospheric balloons using reinforcement learning"](https://www.nature.com/articles/s41586-020-2939-8).

## Installation

Prerequisites: python >= 3.7.

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

## Training an Agent
The set of agents available to train are listed in the [Agent
Registry](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/agents/agent_registry.py).
You can train one of these with the following command:

```
python -m balloon_learning_environment.train \
  --base_dir=/tmp/ble/train \
  --agent=finetune_perciatelli
```

The following are the most useful agents to get started:

*  `station_seeker`: A static controller designed by Loon which has a decent
   performance, and was the baseline against which we compared in our
   [Nature paper](https://www.nature.com/articles/s41586-020-2939-8)
*  `perciatelli44`: The agent we trained for our
   [Nature paper](https://www.nature.com/articles/s41586-020-2939-8).
   It's a frozen agent (so does not do any extra training) but performs quite
    well.
*  `quantile`: A Quantile-based agent in JAX that uses the same
   architecture as Perciatelli44, but starts training from a fresh
   initialization.
*  `finetune_perciatelli`: The same as `quantile`, but reloads the
   `perciatelli44` weights, and is a great way to warm-start agent training.


## Evaluating an Agent

See the [evaluation readme](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/README.md) for instructions on evaluating an agent.

## Giving credit

If you use the Balloon Learning Environment in your work, we ask that you use
the following BibTeX entry:

```
@software{Greaves_Balloon_Learning_Environment_2021,
  author = {Greaves, Joshua and Candido, Salvatore and Dumoulin, Vincent and Goroshin, Ross and Ponda, Sameera S. and Bellemare, Marc G. and Castro, Pablo Samuel},
  month = {12},
  title = {{Balloon Learning Environment}},
  url = {https://github.com/google/balloon-learning-environment},
  version = {1.0.0},
  year = {2021}
}
```

If you use the `ble_wind_field` dataset, you should also cite

```
Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A.,
Muñoz‐Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A.,
Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G.,
Bidlot, J., Bonavita, M., De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M.,
Dragani, R., Flemming, J., Forbes, R., Fuentes, M., Geer, A., Haimberger, L.,
Healy, S., Hogan, R.J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P.,
Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F.,
Villaume, S., Thépaut, J-N. (2017): Complete ERA5: Fifth generation of ECMWF
atmospheric reanalyses of the global climate. Copernicus Climate Change Service
(C3S) Data Store (CDS). (Accessed on 01-04-2021)
```

