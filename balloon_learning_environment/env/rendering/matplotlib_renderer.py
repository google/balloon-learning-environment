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

"""A renderer using matplotlib.

This is not the most efficient renderer, but it is convenient.
"""

from typing import Iterable, Optional, Text, Union

from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env.rendering import renderer
from flax.metrics import tensorboard
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
from mpl_toolkits.mplot3d import art3d
import numpy as np


# TODO(joshgreaves): Add style configurations.
class MatplotlibRenderer(renderer.Renderer):
  """Contains functions for rendering the simulator state with matplotlib."""

  def __init__(self):
    self.reset()

  def reset(self) -> None:
    plt.close('all')
    self.fig = plt.figure(figsize=(8, 6))
    self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
    self._trajectory = list()
    self.hour = 0
    self.charge = 0

  def step(self, state: simulator_data.SimulatorState) -> None:
    balloon_state = state.balloon_state
    self.hour = balloon_state.date_time.hour
    self.charge = balloon_state.battery_soc
    self._trajectory.append(
        np.asarray([balloon_state.x.meters, balloon_state.y.meters,
                    balloon_state.pressure]))

  def render(self,
             mode: Text,
             summary_writer: Optional[tensorboard.SummaryWriter] = None,
             iteration: Optional[int] = None) -> Union[None, np.ndarray, Text]:
    """Renders a frame.

    Args:
      mode: One of `human`, or `rgb_array`. `human` corresponds to rendering
        directly to the screen. `rgb_array` renders to a numpy array and returns
        it. This renderer doesn't support the `ansi` mode.
      summary_writer: If not None and mode == 'tensorboard', will also render
        the image to the tensorboard summary.
      iteration: Iteration number used for writing to tensorboard.

    Returns:
      None or a numpy array of rgb data depending on the mode.
    """
    if mode not in self.render_modes:
      raise ValueError('Unsupported render mode {}. Use one of {}.'.format(
          mode, self.render_modes))

    plt.cla()
    self._plot_data()

    if mode == 'human':
      self.fig.show()
    elif mode == 'rgb_array' or mode == 'tensorboard':
      self.fig.canvas.draw()
      rgb_string = self.fig.canvas.tostring_rgb()
      width, height = self.fig.canvas.get_width_height()
      frame = np.fromstring(
          rgb_string, dtype=np.uint8, sep='').reshape(height, width, -1)
      if mode == 'rgb_array':
        return frame

      if summary_writer is not None and iteration is not None:
        summary_writer.image('Balloon/Path', frame, iteration)
        summary_writer.flush()
        plt.cla()

  @property
  def render_modes(self) -> Iterable[Text]:
    return ['human', 'rgb_array', 'tensorboard']

  def _plot_data(self):

    # TODO(joshgreaves): Longitude/Latitude isn't quite right
    self.ax.set_xlabel('Longitude displacement (km)')
    self.ax.set_ylabel('Latitude displacement (km)')
    self.ax.set_zlabel('Pressure (kPa)')
    self.ax.invert_zaxis()

    data = np.vstack(self._trajectory)
    data /= 1000.0  # m -> km, Pa -> kPa

    self.ax.plot3D(data[:, 0], data[:, 1], data[:, 2], c='k')

    # Draw target areas
    lower_circle = plt.Circle((0.0, 0.0), 50.0, edgecolor='k', ls='--',
                              fill=True, facecolor='blue', zorder=-1, alpha=0.3)
    self.ax.add_patch(lower_circle)
    lower_circle_z = np.max(data[:, 2])
    art3d.pathpatch_2d_to_3d(lower_circle, z=lower_circle_z, zdir='z')
    higher_circle = plt.Circle((0.0, 0.0), 50.0, edgecolor='k', ls='--',
                               fill=True, facecolor='blue', zorder=-1,
                               alpha=0.3)
    self.ax.add_patch(higher_circle)
    upper_circle_z = np.min(data[:, 2])
    art3d.pathpatch_2d_to_3d(higher_circle, z=upper_circle_z, zdir='z')
    # Draw cylinder for target.
    cylinder_x = np.linspace(-50.0, 50.0, 100)
    cylinder_z = np.linspace(lower_circle_z, upper_circle_z, 100)
    mesh_x, mesh_z = np.meshgrid(cylinder_x, cylinder_z)
    mesh_y = np.sqrt(2500 - mesh_x**2)
    self.ax.plot_surface(mesh_x, mesh_y, mesh_z, alpha=0.2, rstride=20,
                         cstride=10)
    self.ax.plot_surface(mesh_x, -mesh_y, mesh_z, alpha=0.2, rstride=20,
                         cstride=10, color='blue')
    self.ax.set_title(f'Hour: {self.hour}, Charge: {self.charge:.2f}',
                      fontsize=32)
