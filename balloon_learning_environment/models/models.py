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

"""Convnience functions for loading models."""

from importlib import resources
import os
from typing import Optional

import gin
import tensorflow as tf

_MODEL_ROOT = 'balloon_learning_environment/models/'
_OFFLINE_SKIES22_RELATIVE_PATH = os.path.join(
    _MODEL_ROOT, 'offlineskies22_decoder.msgpack')
_PERCIATELLI44_RELATIVE_PATH = os.path.join(
    _MODEL_ROOT, 'perciatelli44.pb')


@gin.configurable
def load_offlineskies22(path: Optional[str] = None) -> bytes:
  """Loads offlineskies22 serialized wind VAE parameters.

  There are three places this function looks:
  1. At the path specified, if one is specified.
  2. Under the models package using importlib.resources. It should be
    found there if the code was installed with pip.
  3. Relative to the project root. It should be found there if running
    from a freshly cloned repo.

  Args:
    path: An optional path to load the VAE weights from.

  Returns:
    The serialized VAE weights as bytes.

  Raises:
    ValueError: if a path is specified but the weights can't be loaded.
    RuntimeError: if the weights couldn't be found in any of the
      specified locations.
  """
  # Attempt 1: Load from path, if specified.
  # If a path is specified, we expect it is a good path.
  if path is not None:
    try:
      with tf.io.gfile.GFile(path, 'rb') as f:
        return f.read()
    except tf.errors.NotFoundError:
      raise ValueError(f'offlineskies22 checkpoint not found at {path}')

  # Attempt 2: Load from location expected in the built wheel.
  try:
    with resources.open_binary('balloon_learning_environment.models',
                               'offlineskies22_decoder.msgpack') as f:
      return f.read()
  except FileNotFoundError:
    pass

  # Attempt 3: Load from the path relative to the source root.
  try:
    with tf.io.gfile.GFile(_OFFLINE_SKIES22_RELATIVE_PATH, 'rb') as f:
      return f.read()
  except tf.errors.NotFoundError:
    pass

  raise RuntimeError(
      'Unable to load wind VAE checkpoint from the expected locations.')


@gin.configurable
def load_perciatelli44(path: Optional[str] = None) -> bytes:
  """Loads Perciatelli44.pb as bytes.

  There are three places this function looks:
  1. At the path specified, if one is specified.
  2. Under the models package using importlib.resources. It should be
    found there if the code was installed with pip.
  3. Relative to the project root. It should be found there if running
    from a freshly cloned repo.

  Args:
    path: An optional path to load the VAE weights from.

  Returns:
    The serialized VAE weights as bytes.

  Raises:
    ValueError: if a path is specified but the weights can't be loaded.
    RuntimeError: if the weights couldn't be found in any of the
      specified locations.
  """
  # Attempt 1: Load from path, if specified.
  # If a path is specified, we expect it is a good path.
  if path is not None:
    try:
      with tf.io.gfile.GFile(path, 'rb') as f:
        return f.read()
    except tf.errors.NotFoundError:
      raise ValueError(f'perciatelli44 checkpoint not found at {path}')

  # Attempt 2: Load from location expected in the built wheel.
  try:
    with resources.open_binary('balloon_learning_environment.models',
                               'perciatelli44.pb') as f:
      return f.read()
  except FileNotFoundError:
    pass

  # Attempt 3: Load from the path relative to the source root.
  try:
    with tf.io.gfile.GFile(_PERCIATELLI44_RELATIVE_PATH, 'rb') as f:
      return f.read()
  except FileNotFoundError:
    pass

  raise RuntimeError(
      'Unable to load Perciatelli44 checkpoint from the expected locations.')
