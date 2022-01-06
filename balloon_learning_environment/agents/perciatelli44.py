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

"""A frozen Perciatelli44 agent."""

from typing import Sequence

from balloon_learning_environment.agents import agent
from balloon_learning_environment.models import models
import numpy as np
import tensorflow as tf


def load_perciatelli_session() -> tf.compat.v1.Session:
  serialized_perciatelli = models.load_perciatelli44()

  sess = tf.compat.v1.Session()
  graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(serialized_perciatelli)

  tf.compat.v1.import_graph_def(graph_def)
  return sess


class Perciatelli44(agent.Agent):
  """Perciatelli44 Agent.

  This is the agent which was reported as state of the art in
  "Autonomous navigation of stratospheric balloons using reinforcement
  learning" (Bellemare, Candido, Castro, Gong, Machado, Moitra, Ponda,
  and Wang, 2020).

  This agent has its weights frozen, and is intended for comparison in
  evaluation, not for retraining.
  """

  def __init__(self, num_actions: int, observation_shape: Sequence[int]):
    super(Perciatelli44, self).__init__(num_actions, observation_shape)

    if num_actions != 3:
      raise ValueError('Perciatelli44 only supports 3 actions.')
    if list(observation_shape) != [1099]:
      raise ValueError('Perciatelli44 only supports 1099 dimensional input.')

    # TODO(joshgreaves): It would be nice to use the saved_model API
    # for loading the Perciatelli graph.
    # TODO(joshgreaves): We wanted to avoid a dependency on TF, but adding
    # this to the agent registry makes TF a necessity.
    self._sess = load_perciatelli_session()
    self._action = self._sess.graph.get_tensor_by_name('sleepwalk_action:0')
    self._q_vals = self._sess.graph.get_tensor_by_name('q_values:0')
    self._observation = self._sess.graph.get_tensor_by_name('observation:0')

  def begin_episode(self, observation: np.ndarray) -> int:
    observation = observation.reshape((1, 1099))
    q_vals = self._sess.run(self._q_vals,
                            feed_dict={self._observation: observation})
    return np.argmax(q_vals).item()

  def step(self, reward: float, observation: np.ndarray) -> int:
    observation = observation.reshape((1, 1099))
    q_vals = self._sess.run(self._q_vals,
                            feed_dict={self._observation: observation})
    return np.argmax(q_vals).item()

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    pass
