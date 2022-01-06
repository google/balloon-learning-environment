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

"""Tests for vae."""

from absl.testing import absltest
from balloon_learning_environment.generative import vae
import jax


class VaeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.input_shape = (8,)
    self.sample_input = jax.random.normal(self.key, self.input_shape)

  def test_encoder_computes_valid_mean_and_logvar(self):
    num_latents = 10
    encoder = vae.Encoder(num_latents=num_latents)
    params = encoder.init(self.key, self.sample_input)

    mean, logvar = encoder.apply(params, self.sample_input)

    # Since mean and logvar are valid in (-inf, inf), just check the shape.
    self.assertEqual(mean.shape, (num_latents,))
    self.assertEqual(logvar.shape, (num_latents,))

  def test_decoder_computes_valid_wind_field(self):
    num_latents = 10
    sample_latents = jax.random.normal(self.key, (num_latents,))

    decoder = vae.Decoder()
    params = decoder.init(self.key, sample_latents)

    reconstructed_wind_field = decoder.apply(params, sample_latents)

    # Since reconstruction is valid in (-inf, inf), just check the shape.
    self.assertEqual(reconstructed_wind_field.shape,
                     vae.FieldShape().grid_shape())

  def test_vae_computes_valid_wind_field(self):
    num_latents = 10
    vae_def = vae.WindFieldVAE(num_latents=num_latents)
    params = vae_def.init(self.key, self.sample_input, self.key)

    vae_output = vae_def.apply(params, self.sample_input, self.key)

    self.assertEqual(vae_output.reconstruction.shape,
                     vae.FieldShape().grid_shape())
    self.assertEqual(vae_output.encoder_output.mean.shape, (num_latents,))
    self.assertEqual(vae_output.encoder_output.logvar.shape, (num_latents,))


if __name__ == '__main__':
  absltest.main()
