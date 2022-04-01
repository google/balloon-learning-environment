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

"""Common transforms."""

import typing
from typing import Union

import numpy as np


def _contains_negative_values(x: Union[float, np.ndarray]) -> bool:
  if isinstance(x, np.ndarray):
    return (x < 0).any()
  else:
    return x < 0


@typing.overload
def linear_rescale_with_extrapolation(x: np.ndarray,
                                      vmin: float,
                                      vmax: float) -> np.ndarray:
  ...


@typing.overload
def linear_rescale_with_extrapolation(x: float,
                                      vmin: float,
                                      vmax: float) -> float:
  ...


def linear_rescale_with_extrapolation(x,
                                      vmin,
                                      vmax):
  """Returns x normalized between [vmin, vmax], with possible extrapolation."""
  if vmax <= vmin:
    raise ValueError('Interval must be such that vmax > vmin.')
  else:
    return (x - vmin) / (vmax - vmin)


def undo_linear_rescale_with_extrapolation(x: float, vmin: float,
                                           vmax: float) -> float:
  """Computes the input of linear_rescale_with_extrapolation given output."""
  if vmax <= vmin:
    raise ValueError('Interval must be such that vmax > vmin.')
  return vmin + x * (vmax - vmin)


def linear_rescale_with_saturation(x: float, vmin: float, vmax: float) -> float:
  """Returns x normalized in [0, 1]."""
  y = linear_rescale_with_extrapolation(x, vmin, vmax)
  return np.clip(y, 0.0, 1.0).item()


@typing.overload
def squash_to_unit_interval(x: np.ndarray, constant: float) -> np.ndarray:
  ...


@typing.overload
def squash_to_unit_interval(x: float, constant: float) -> float:
  ...


def squash_to_unit_interval(x, constant):
  """Scales non-negative x to be in range [0, 1], with a squash."""
  if constant <= 0:
    raise ValueError('Squash constant must be greater than zero.')
  if _contains_negative_values(x):
    raise ValueError('Squash can only be performed on non-negative values.')
  return x / (x + constant)


def undo_squash_to_unit_interval(x: float, constant: float) -> float:
  """Computes the input value of squash_to_unit_interval given the output."""
  if constant <= 0:
    raise ValueError('Squash constant must be greater than zero.')
  if 0 > x >= 1:
    raise ValueError('Undo squash can only be performed on a value in [0, 1).')
  return (x * constant) / (1 - x)
