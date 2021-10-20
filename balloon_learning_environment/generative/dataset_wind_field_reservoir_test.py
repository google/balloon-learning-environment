# coding=utf-8
# Copyright 2021 The Balloon Learning Environment Authors.
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

"""Tests for dataset_wind_field_reservoir."""

from absl.testing import absltest
from balloon_learning_environment.generative import dataset_wind_field_reservoir
import jax.numpy as jnp


def _make_dataset() -> jnp.ndarray:
  train = 0.5 * jnp.ones(shape=(33, 2, 3, 4))
  test = 0.2 * jnp.ones(shape=(10, 2, 3, 4))
  return jnp.concatenate([train, test], axis=0)


class DatasetWindFieldReservoirTest(absltest.TestCase):

  def test_reservoir_returns_eval_for_eval(self):
    reservoir = dataset_wind_field_reservoir.DatasetWindFieldReservoir(
        data=_make_dataset(), eval_batch_size=10)
    self.assertTrue(jnp.allclose(reservoir.get_eval_batch(),
                                 0.2 * jnp.ones(shape=(10, 2, 3, 4))))

  def test_reservoir_returns_train_for_train(self):
    reservoir = dataset_wind_field_reservoir.DatasetWindFieldReservoir(
        data=_make_dataset(), eval_batch_size=10)
    self.assertTrue(jnp.allclose(reservoir.get_batch(batch_size=33),
                                 0.5 * jnp.ones(shape=(33, 2, 3, 4))))


if __name__ == '__main__':
  absltest.main()
