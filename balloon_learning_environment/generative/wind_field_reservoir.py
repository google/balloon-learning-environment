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

"""An interface to datasets of wind fields used for training a VAE model."""

import abc
import jax.numpy as jnp


class WindFieldReservoir(abc.ABC):
  """Abstract class for wind datasets for training."""

  @abc.abstractmethod
  def get_batch(self, batch_size: int) -> jnp.ndarray:
    """Returns fields used for training.

    Args:
      batch_size: The number of fields.

    Returns:
      A jax.numpy array that is batch_size x wind field dimensions (see vae.py).
    """

  @abc.abstractmethod
  def get_eval_batch(self) -> jnp.ndarray:
    """Returns fields used for eval.

    Returns:
      A jax.numpy array that is batch_size x wind field dimensions (see vae.py).
    """
