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

"""Tests for dopamine_utils."""

import os.path as osp
import pickle
from unittest import mock
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from balloon_learning_environment.agents import dopamine_utils
import tensorflow as tf


class DopamineUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_subdir = self.create_tempdir().full_path

  @parameterized.named_parameters(
      dict(testcase_name='no_bundle', bundle=None),
      dict(testcase_name='with_bundle', bundle={'foo': 'test'}))
  @mock.patch.object(pickle, 'dump', autospec=True)
  @mock.patch.object(tf.io.gfile, 'GFile', autospec=False)
  @mock.patch.object(logging, 'warning', autospec=True)
  def test_save_checkpoint(self, mock_logger, mock_gfile, mock_pickle, bundle):
    checkpoint_dir = '/tmp/test'
    iteration_number = 5
    bundle_fn = lambda x, y: bundle
    dopamine_utils.save_checkpoint(checkpoint_dir, iteration_number,
                                   bundle_fn)
    logging_calls = 1 if bundle is None else 0
    self.assertEqual(mock_logger.call_count, logging_calls)
    if bundle is None:
      return

    self.assertEqual(mock_gfile.call_count, 1)
    self.assertEqual(
        mock_gfile.call_args[0][0],
        f'{checkpoint_dir}/checkpoint_{iteration_number:05d}.pkl')
    self.assertEqual(mock_gfile.call_args[0][1], 'w')
    self.assertEqual(mock_pickle.call_count, 1)
    self.assertEqual(mock_pickle.call_args[0][0], bundle)

  @parameterized.named_parameters(
      dict(testcase_name='no_checkpoint_no_unbundle',
           checkpoint_exists=False, unbundle=False),
      dict(testcase_name='no_checkpoint_unbundle',
           checkpoint_exists=False, unbundle=True),
      dict(testcase_name='checkpoint_no_unbundle',
           checkpoint_exists=True, unbundle=False),
      dict(testcase_name='checkpoint_unbundle',
           checkpoint_exists=True, unbundle=True))
  @mock.patch.object(pickle, 'load', autospec=True)
  @mock.patch.object(tf.io.gfile, 'GFile', autospec=False)
  @mock.patch.object(tf.io.gfile, 'exists', autospec=False)
  @mock.patch.object(logging, 'warning', autospec=True)
  def test_load_checkpoint(self, mock_logger, mock_exists, mock_gfile,
                           mock_pickle, checkpoint_exists, unbundle):
    checkpoint_dir = '/tmp/test'
    iteration_number = 5
    mock_exists.return_value = checkpoint_exists
    bundle = {'foo': 'test'}
    mock_pickle.return_value = bundle
    unbundle_fn = mock.MagicMock(return_value=unbundle)
    dopamine_utils.load_checkpoint(checkpoint_dir, iteration_number,
                                   unbundle_fn)

    self.assertEqual(mock_exists.call_count, 1)
    logger_count = 1 if (not checkpoint_exists or not unbundle) else 0
    self.assertEqual(mock_logger.call_count, logger_count)
    if not checkpoint_exists:
      return

    self.assertEqual(mock_gfile.call_count, 1)
    self.assertEqual(
        mock_gfile.call_args[0][0],
        f'{checkpoint_dir}/checkpoint_{iteration_number:05d}.pkl')
    self.assertEqual(mock_gfile.call_args[0][1], 'rb')
    self.assertEqual(mock_pickle.call_count, 1)

  def test_get_latest_checkpoint_with_invalid_dir(self):
    self.assertEqual(
        -1, dopamine_utils.get_latest_checkpoint('/does/not/exist'))

  def test_get_latest_checkpoint_with_empty_dir(self):
    self.assertEqual(
        -1, dopamine_utils.get_latest_checkpoint(self._test_subdir))

  def test_get_latest_checkpoint(self):
    filename = osp.join(self._test_subdir, 'checkpoint_00123.pkl')
    _ = self.create_tempfile(filename).full_path
    self.assertEqual(123,
                     dopamine_utils.get_latest_checkpoint(self._test_subdir))

  def test_clean_up_old_checkpoints_with_empty_dir(self):
    dopamine_utils.clean_up_old_checkpoints(self._test_subdir, 10)

  @parameterized.parameters((None), (5), (100), (200))
  def test_clean_up_old_checkpoints(self, checkpoint_duration):
    last_checkpoint = 100
    for i in range(last_checkpoint):
      filename = dopamine_utils._make_checkpoint_filename(self._test_subdir,
                                                          i)
      _ = self.create_tempfile(filename)

    # First make sure all the files were created.
    self.assertLen(tf.io.gfile.glob(osp.join(self._test_subdir, '*')),
                   last_checkpoint)

    if checkpoint_duration is None:
      for i in range(1, 101):
        dopamine_utils.clean_up_old_checkpoints(self._test_subdir, i)
      checkpoint_duration = 50  # Default value
    else:
      for i in range(1, 101):
        dopamine_utils.clean_up_old_checkpoints(
            self._test_subdir, i, checkpoint_duration=checkpoint_duration)

    # Ensure only 50 checkpoints remain (the default).
    remaining_checkpoints = tf.io.gfile.glob(osp.join(self._test_subdir, '*'))
    self.assertLen(remaining_checkpoints,
                   min(checkpoint_duration, last_checkpoint))
    # Verify that these are the last checkpoint_duration checkpoints.
    for i in range(max(0, last_checkpoint - checkpoint_duration),
                   last_checkpoint):
      filename = osp.join(self._test_subdir, f'checkpoint_{i:05d}.pkl')
      self.assertIn(filename, remaining_checkpoints)

if __name__ == '__main__':
  absltest.main()
