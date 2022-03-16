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

"""An interface for sampling grid wind fields."""

import abc
import datetime as dt

from balloon_learning_environment.generative import vae
from jax import numpy as jnp
import numpy as np


class GridWindFieldSampler(abc.ABC):

  @property
  @abc.abstractmethod
  def field_shape(self) -> vae.FieldShape:
    """Gets the field shape of wind fields sampled by this class."""

  @abc.abstractmethod
  def sample_field(self,
                   key: jnp.ndarray,
                   date_time: dt.datetime) -> np.ndarray:
    """Samples a wind field and returns it as a numpy array.

    Args:
      key: A PRNGKey to use for sampling.
      date_time: The date_time of the begining on the wind field.
    """
