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

"""Tests for models."""

from importlib import resources
import io
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.models import models
import tensorflow as tf


class ModelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='openskies22', load_fn=models.load_offlineskies22),
      dict(testcase_name='perciatelli44', load_fn=models.load_offlineskies22))
  def test_load_with_specified_path_loads_data(self, load_fn):
    # Write some fake data in a tmpdir.
    tmpfile = self.create_tempfile()
    fake_content = b'fake content from specified path'
    with open(tmpfile, 'wb') as f:
      f.write(fake_content)

    result = load_fn(tmpfile.full_path)

    self.assertEqual(result, fake_content)

  @parameterized.named_parameters(
      dict(testcase_name='openskies22', load_fn=models.load_offlineskies22),
      dict(testcase_name='perciatelli44', load_fn=models.load_offlineskies22))
  def test_load_with_wrong_path_fails(self, load_fn):
    with self.assertRaises(ValueError):
      load_fn('this_is_not_a_valid_path')

  @parameterized.named_parameters(
      dict(testcase_name='openskies22', load_fn=models.load_offlineskies22),
      dict(testcase_name='perciatelli44', load_fn=models.load_offlineskies22))
  @mock.patch.object(resources, 'open_binary', autospec=True)
  def test_load_uses_importlib_if_no_path_is_specified(self,
                                                       mock_open,
                                                       load_fn):
    fake_content = b'fake content from importlib.resources'
    mock_open.return_value = io.BytesIO(fake_content)

    result = load_fn()

    mock_open.assert_called_once()
    self.assertEqual(result, fake_content)

  @parameterized.named_parameters(
      dict(testcase_name='openskies22', load_fn=models.load_offlineskies22),
      dict(testcase_name='perciatelli44', load_fn=models.load_offlineskies22))
  @mock.patch.object(tf.io.gfile, 'GFile', autospec=True)
  def test_load_uses_default_path_as_last_resort(self,
                                                 mock_gfile,
                                                 load_fn):
    fake_content = b'fake content from default path'
    mock_gfile.return_value = io.BytesIO(fake_content)

    result = load_fn()

    mock_gfile.assert_called_once()
    self.assertEqual(result, fake_content)

  @parameterized.named_parameters(
      dict(testcase_name='openskies22', load_fn=models.load_offlineskies22),
      dict(testcase_name='perciatelli44', load_fn=models.load_offlineskies22))
  def test_load_raises_runtime_error_if_not_found(self,
                                                  load_fn):
    with self.assertRaises(RuntimeError):
      load_fn()

if __name__ == '__main__':
  absltest.main()
