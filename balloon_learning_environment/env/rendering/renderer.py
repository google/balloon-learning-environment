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

"""Abstract base class for renderers."""

import abc
from typing import Iterable, Optional, Text, Union

from balloon_learning_environment.env import simulator_data
from flax.metrics import tensorboard
import numpy as np


class Renderer(abc.ABC):
  """A renderer object for rendering the simulator state."""

  @abc.abstractmethod
  def reset(self) -> None:
    pass

  @abc.abstractmethod
  def step(self, state: simulator_data.SimulatorState) -> None:
    pass

  @abc.abstractmethod
  def render(self,
             mode: Text,
             summary_writer: Optional[tensorboard.SummaryWriter] = None,
             iteration: Optional[int] = None) -> Union[None, np.ndarray, Text]:
    """Renders a frame.

    Args:
      mode: A string specifying the mode. Default gym render modes are `human`,
        `rgb_array`, and `ansi`. However, a renderer may specify additional
        render modes beyond this. `human` corresponds to rendering directly to
        the screen. `rgb_array` renders to a numpy array and returns it. `ansi`
        renders to a string or StringIO object.
      summary_writer: If not None, will also render the image to the tensorboard
        summary.
      iteration: Iteration number used for writing to tensorboard.

    Returns:
      None, a numpy array of rgb data, or a Text object, depending on the mode.
    """
    pass

  @property
  @abc.abstractmethod
  def render_modes(self) -> Iterable[Text]:
    pass
