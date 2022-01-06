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

"""Tests for balloon_learning_environment.agents.networks."""

from absl.testing import absltest
from balloon_learning_environment.agents import networks
import gin
import jax
import jax.numpy as jnp


class NetworksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._num_actions = 4
    self._observation_shape = (6, 7)
    self._example_state = jnp.zeros(self._observation_shape)
    gin.bind_parameter('MLPNetwork.num_layers', 1)
    gin.bind_parameter('MLPNetwork.hidden_units', 256)

  def _create_network(self):
    self._network_def = networks.MLPNetwork(num_actions=self._num_actions)

  def test_default_network_parameters(self):
    self._create_network()
    self.assertEqual(self._num_actions, self._network_def.num_actions)
    self.assertEqual(1, self._network_def.num_layers)
    self.assertEqual(256, self._network_def.hidden_units)
    network_params = self._network_def.init(jax.random.PRNGKey(0),
                                            self._example_state)
    self.assertIn('Dense_0', network_params['params'])

  def test_custom_network(self):
    num_layers = 5
    hidden_units = 64
    gin.bind_parameter('MLPNetwork.num_layers', num_layers)
    gin.bind_parameter('MLPNetwork.hidden_units', hidden_units)
    self._create_network()
    self.assertEqual(num_layers, self._network_def.num_layers)
    self.assertEqual(hidden_units, self._network_def.hidden_units)
    network_params = self._network_def.init(jax.random.PRNGKey(0),
                                            self._example_state)
    self.assertIn('Dense_0', network_params['params'])
    for i in range(num_layers - 1):
      self.assertIn(f'Dense_{i}', network_params['params'])

  def test_call_network(self):
    self._create_network()
    network_params = self._network_def.init(jax.random.PRNGKey(0),
                                            self._example_state)
    # All zeros in should produce all zeros out, since the default initializer
    # for bias is all zeros.
    zeros_out = self._network_def.apply(network_params,
                                        jnp.zeros_like(self._example_state))
    self.assertTrue(jnp.array_equal(zeros_out, jnp.zeros(self._num_actions)))
    # All ones in should produce something that is non-zero at the output, since
    # we are using Glorot initiazilation.
    ones_out = self._network_def.apply(network_params,
                                       jnp.ones_like(self._example_state))
    self.assertFalse(jnp.array_equal(ones_out, jnp.zeros(self._num_actions)))


if __name__ == '__main__':
  absltest.main()
