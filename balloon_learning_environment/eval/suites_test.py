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

"""Tests for suites."""

from absl.testing import absltest
from balloon_learning_environment.eval import suites


class SuitesTest(absltest.TestCase):

  def test_get_eval_suite_is_successful_for_valid_name(self):
    eval_suite = suites.get_eval_suite('big_eval')

    self.assertLen(eval_suite.seeds, 10_000)

  def test_get_eval_suite_raises_error_for_invalid_name(self):
    with self.assertRaises(ValueError):
      suites.get_eval_suite('invalid name')


if __name__ == '__main__':
  absltest.main()
