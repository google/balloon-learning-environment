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

r"""Learns a generative model to mimick wind fields in the stratosphere.

"""

import functools
from os import path as osp
import time
from typing import Callable, Sequence

from absl import app
from absl import flags
from absl import logging
from balloon_learning_environment.generative import dataset_wind_field_reservoir
from balloon_learning_environment.generative import vae
from balloon_learning_environment.utils import wind
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
import jax.numpy as jnp



flags.DEFINE_string('checkpoint_directory', '',
                    'Directory to save training snapshots.')

flags.DEFINE_string(
    'bootstrap_directory', '',
    'Directory to restart training from the checkpoint of another training run.'
)

flags.DEFINE_string('offline_winds_dataset_path', '',
                    'Path to an offline dataset.')

flags.DEFINE_multi_string('gin_bindings', [],
                          'Gin bindings to override default values.')

FLAGS = flags.FLAGS


# TODO(joshgreaves): Move schedules elsewhere.
# TODO(joshgreaves): Make tests for these.
@gin.configurable(allowlist=['value'])
def constant_schedule(unused_idx: int, value: float = gin.REQUIRED) -> float:
  """A constant schedule.

  Args:
    unused_idx: The epoch index. Not used by this schedule.
    value: The constant value to return. Should be set by gin.

  Returns:
    value.
  """
  return value


@gin.configurable(denylist=['idx'])
def step_schedule(idx: int,
                  *,
                  start_value: float = gin.REQUIRED,
                  increment: float = gin.REQUIRED,
                  frequency: int = gin.REQUIRED,
                  max_val: float = gin.REQUIRED) -> float:
  """A step schedule.

  Args:
    idx: The epoch index.
    start_value: The value the schedule will return at epoch 0.
    increment: The step size. This is multiplied to start_value at regular
      intervals.
    frequency: How often to increment the value from the schedule, in epochs.
    max_val: The maximum value to return from the schedule.

  Returns:
    A value that is increased at regular intervals.
  """
  return min(start_value * (increment**(idx // frequency)), max_val)


def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def mean_square_error(reconstructed, original):
  return jnp.sum((reconstructed - original)**2)


@jax.jit
def train_step(optimizer, batch, rng, kl_weight: float = 512.0):
  """A unit of VAE training work.

  Args:
    optimizer: The Flax optimizer.
    batch: A batch of training data (wind fields).
    rng: Jax random number generator.
    kl_weight: Coefficient balancing MSE and KLD losses in final scalar loss.

  Returns:
    State of the optimizer after a round of training and the training loss.
  """

  def _loss_fn(params, x, rng):
    """Loss function for training.

    Args:
      params: Neural network weights from the Flax optimizer.
      x: A training example.
      rng: Pseudorandom number generator.

    Returns:
      The (scalar) loss value.
    """
    reconstructed, (mean, logvar), sigma = vae.model().apply(params, x, rng)
    mse = mean_square_error(reconstructed, x)
    kld = kl_divergence(mean, logvar)

    combined_loss = 0.5 / sigma**2 * mse + jnp.log(
        sigma * jnp.sqrt(2 * jnp.pi)) + kl_weight * kld
    return combined_loss, (mse, kld, sigma)

  rngs = jnp.stack(jax.random.split(rng, num=batch.shape[0]))
  grad_fn = jax.vmap(jax.value_and_grad(_loss_fn, has_aux=True),
                     in_axes=(None, 0, 0))
  (combined_loss, aux_losses), grads = grad_fn(optimizer.target, batch, rngs)
  mse, kld, sigma = aux_losses

  loss = jnp.mean(combined_loss)
  mse = jnp.mean(mse)
  kld = jnp.mean(kld)
  grad = jax.tree_map(functools.partial(jnp.mean, axis=0), grads)

  optimizer = optimizer.apply_gradient(grad)

  return optimizer, loss, mse, kld, sigma


@jax.jit
def evaluation(optimizer, fields, prng_initial_seed=0):
  """Evaluate the current solution the optimizer has found with fixed fields.

  Args:
    optimizer: The Flax optimizer.
    fields: A batch of training data (wind fields).
    prng_initial_seed: Seed used by the PRNG.

  Returns:
    A dictionary with stats about losses.
  """

  def _eval_field(params, field, key):
    """Function to eval on a particular field."""

    result: vae.VAEOutput = vae.WindFieldVAE().apply(params, field, key)

    mse_loss = mean_square_error(result.reconstruction, field)
    kld_loss = kl_divergence(result.encoder_output.mean,
                             result.encoder_output.logvar)

    return result.reconstruction, result.encoder_output.mean, mse_loss, kld_loss

  eval_all_fields = jax.vmap(_eval_field, in_axes=(None, 0, 0))

  # NOTE(scandido): Use the same prng in eval to remove a source of variation
  # from the metrics.
  rng = jax.random.PRNGKey(prng_initial_seed)
  keys = jax.random.split(rng, num=fields.shape[0])

  reconstructed_fields, latents, mse_loss, kld_loss = eval_all_fields(
      optimizer.target, fields, keys)

  mean_speed_reconstructed = jnp.array(
      jax.vmap(wind.mean_speed_in_wind_field)(reconstructed_fields))
  mean_speed_original = jnp.array(
      jax.vmap(wind.mean_speed_in_wind_field)(fields))
  mean_speed_differential = mean_speed_original - mean_speed_reconstructed

  metrics = {
      'kld': kld_loss.mean(),
      'mse': mse_loss.mean(),
      'mean_speed_reconstructed': mean_speed_reconstructed.mean(),
      'mean_speed_original': mean_speed_original.mean(),
      'mean_speed_differential': mean_speed_differential.mean(),
  }
  return metrics, jnp.stack(reconstructed_fields), jnp.stack(latents)


@gin.configurable
def train(num_batches_per_epoch: int = 200,
          num_epochs: float = 1e5,
          learning_rate: float = 1e-5,
          *,
          kl_schedule: Callable[[int], float] = gin.REQUIRED) -> None:
  """Training loop for learning the VAE.

  Args:
    num_batches_per_epoch: Number of training rounds per evaluation of the
      solution.
    num_epochs: Number of epochs to train before terminating the program.
    learning_rate: Learning rate used by the optimizer for gradient descent.
    kl_schedule: A function that takes an epoch idx and returns "beta", the
      kl coefficient to use for training.
  """

  # NOTE(scandido): Use the same prng in eval to remove a source of variation
  # Start by fetching a batch of wind fields for evaluation of how training is
  # going. We'll want to keep these the same throughout the training process so
  # we pull the data into an array and hang on to it.
  #
  # TODO(scandido): In order to compare different parameter sets we're going to
  # have to stabilize the wind fields we use across different workers.
  reservoir = dataset_wind_field_reservoir.DatasetWindFieldReservoir(
      data=FLAGS.offline_winds_dataset_path, eval_batch_size=10)
  eval_fields = reservoir.get_eval_batch()

  rng = jax.random.PRNGKey(int(time.time() * 1000))
  rng, key = jax.random.split(rng)

  params = vae.model().init(key,
                            jnp.ones(vae.FieldShape().output_length()),
                            rng)
  optimizer = optim.Adam(learning_rate=learning_rate).create(params)
  optimizer = jax.device_put(optimizer)

  if FLAGS.checkpoint_directory:
    # NOTE(scandido): Passes through optimizer if there is no checkpoint.
    optimizer = checkpoints.restore_checkpoint(
        FLAGS.checkpoint_directory, optimizer)

  # If we have a fresh optimizer (not loading from checkpoint) and a bootstrap
  # directory is specified we attempt to bootstrap from a previous training run.
  if FLAGS.bootstrap_directory and optimizer.state.step == 0:
    # NOTE(scandido): Passes through optimizer if there is no bootstrap
    # checkpoint.
    optimizer = checkpoints.restore_checkpoint(
        FLAGS.bootstrap_directory, optimizer)

  epoch = optimizer.state.step // num_batches_per_epoch + 1


  if FLAGS.checkpoint_directory:
    summary_writer = tensorboard.SummaryWriter(FLAGS.checkpoint_directory)
  else:
    summary_writer = None

  for epoch in range(epoch, int(num_epochs)):
    kl_coefficient = kl_schedule(epoch)
    losses = []
    mses = []
    klds = []
    sigmas = []
    batch_wind_speeds = []
    for _ in range(num_batches_per_epoch):
      batch = reservoir.get_batch(batch_size=64)

      rng, key = jax.random.split(rng)
      optimizer, loss, mse, kld, sigma = train_step(
          optimizer, batch, key, kl_coefficient)
      losses.append(loss)
      mses.append(mse)
      klds.append(kld)
      sigmas.append(sigma)
      batch_wind_speeds.append(
          jnp.array(jax.vmap(wind.mean_speed_in_wind_field)(batch)).mean())

    metrics, _, _ = evaluation(optimizer, eval_fields)

    avg_training_loss = jnp.array(losses).mean()
    avg_mse_loss = jnp.array(mses).mean()
    avg_kld_loss = jnp.array(klds).mean()
    avg_batch_wind_speed = jnp.array(batch_wind_speeds).mean()
    avg_sigma = jnp.array(sigma).mean()

    mse_series.create_measurement(metrics['mse'], epoch)
    kld_series.create_measurement(metrics['kld'], epoch)
    loss_series.create_measurement(avg_training_loss, epoch)

    logging.info(
        'epoch: %d, mse: %.4f, kld: %.4f, train loss: %.4f, sigma: %.4f', epoch,
        metrics['mse'], metrics['kld'], avg_training_loss, avg_sigma)

    if summary_writer:
      summary_writer.scalar('train/loss', avg_training_loss, epoch)
      summary_writer.scalar('train/mse', avg_mse_loss, epoch)
      summary_writer.scalar('train/kld', avg_kld_loss, epoch)
      summary_writer.scalar('train/mean_speed', avg_batch_wind_speed, epoch)
      summary_writer.scalar('train/kl_coefficient', kl_coefficient, epoch)
      summary_writer.scalar('train/sigma', avg_sigma, epoch)
      summary_writer.scalar('eval/mse', metrics['mse'], epoch)
      summary_writer.scalar('eval/kld', metrics['kld'], epoch)
      summary_writer.scalar('eval/mean_speed_reconstructed',
                            metrics['mean_speed_reconstructed'], epoch)
      summary_writer.scalar('eval/mean_speed_original',
                            metrics['mean_speed_original'], epoch)
      summary_writer.scalar('eval/mean_speed_differential',
                            metrics['mean_speed_differential'], epoch)

    if FLAGS.checkpoint_directory:
      checkpoints.save_checkpoint(
          FLAGS.checkpoint_directory,
          optimizer,
          epoch,
          keep=1)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [], bindings=FLAGS.gin_bindings, skip_unknown=False)


  train()


if __name__ == '__main__':
  app.run(main)
