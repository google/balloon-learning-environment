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

"""A wind field reservoir that loads a dataset from a file."""

import pickle
from typing import Union

from absl import logging
from balloon_learning_environment.generative import wind_field_reservoir
import jax
import jax.numpy as jnp
import tensorflow as tf


class DatasetWindFieldReservoir(wind_field_reservoir.WindFieldReservoir):
  """Retrieves wind fields from an in-memory datastore."""

  def __init__(self,
               data: Union[str, jnp.ndarray],
               eval_batch_size: int = 10,
               rng_seed=0):
    self.eval_batch_size = eval_batch_size

    if isinstance(data, str):
      # TODO(scandido): We need to update this to load a single file, with no
      # assumed directory/file structure hardcoded.
      def _get_shard(i: int):
        fn = f'{data}/batch{i:04d}.pickle'
        with tf.io.gfile.GFile(fn, 'rb') as f:
          arr = pickle.load(f)
        return arr

      dataset_shards = []
      for i in range(200):
        dataset_shards.append(_get_shard(i))
        logging.info('Loaded shard %d', i)
      data = jnp.concatenate(dataset_shards, axis=0)

    self.dataset = data
    self._rng = jax.random.PRNGKey(rng_seed)

  def get_batch(self, batch_size: int) -> jnp.ndarray:
    """Returns fields used for training.

    Args:
      batch_size: The number of fields.

    Returns:
      A jax.numpy array that is batch_size x wind field dimensions (see vae.py).
    """

    self._rng, key = jax.random.split(self._rng)
    samples = jax.random.choice(
        key,
        self.dataset.shape[0] - self.eval_batch_size,
        shape=(batch_size,),
        replace=False)
    return self.dataset[samples, ...]

  def get_eval_batch(self) -> jnp.ndarray:
    """Returns fields used for eval.

    Returns:
      A jax.numpy array that is eval_batch_size x wind field dimensions (see
      vae.py).
    """

    return self.dataset[-self.eval_batch_size:, ...]
