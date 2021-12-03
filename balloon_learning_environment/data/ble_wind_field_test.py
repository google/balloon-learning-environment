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

"""Tests for ble_wind_field."""

import os
from unittest import mock

from balloon_learning_environment.data import ble_wind_field
import tensorflow_datasets as tfds


class BLEWindFieldTest(tfds.testing.DatasetBuilderTestCase):
  DATASET_CLASS = ble_wind_field.BLEWindField
  SPLITS = {'train': 4}
  EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'placeholder')

  def setUp(self):
    super().setUp()
    # Patch BLEWindField's GCS_URL and GCS_FILENAME class attributes to
    # point to the placeholder Zarr array instead.
    module_name = self.DATASET_CLASS.__module__
    class_name = self.DATASET_CLASS.__name__
    patchers = [
        mock.patch(f'{module_name}.{class_name}.GCS_URL', self.EXAMPLE_DIR),
        mock.patch(f'{module_name}.{class_name}.GCS_FILENAME', 'array.zarr'),
    ]
    for patcher in patchers:
      patcher.start()
    self.patchers.extend(patchers)


if __name__ == '__main__':
  tfds.testing.test_main()
