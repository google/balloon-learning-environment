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

"""Tests for balloon_learning_environment.metrics.collector."""

import os.path as osp

from absl import flags
from absl.testing import absltest
from balloon_learning_environment.metrics import collector


# A simple subclass that implements the abstract methods.
class SimpleCollector(collector.Collector):

  def get_name(self) -> str:
    return 'simple'

  def pre_training(self) -> None:
    pass

  def begin_episode(self) -> None:
    pass

  def step(self, unused_statistics) -> None:
    pass

  def end_episode(self, unused_statistics) -> None:
    pass

  def end_training(self) -> None:
    pass


class CollectorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._na = 5
    self._tmpdir = flags.FLAGS.test_tmpdir

  def test_instantiate_abstract_class(self):
    # It is not possible to instantiate Collector as it has abstract methods.
    with self.assertRaises(TypeError):
      collector.Collector(self._tmpdir, self._na, 'fail')

  def test_valid_subclass(self):
    simple_collector = SimpleCollector(self._tmpdir, self._na, 0)
    self.assertEqual(simple_collector._base_dir,
                     osp.join(self._tmpdir, 'metrics/simple'))
    self.assertEqual(self._na, simple_collector._num_actions)
    self.assertTrue(osp.exists(simple_collector._base_dir))

  def test_valid_subclass_with_no_basedir(self):
    simple_collector = SimpleCollector(None, self._na, 0)
    self.assertIsNone(simple_collector._base_dir)
    self.assertEqual(self._na, simple_collector._num_actions)


if __name__ == '__main__':
  absltest.main()
