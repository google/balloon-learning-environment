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

"""A collection of evaluation suites."""

from typing import List

from balloon_learning_environment.eval import eval_lib


_eval_suites = dict()


_eval_suites['big_eval'] = eval_lib.EvaluationSuite(list(range(10_000)), 960)
_eval_suites['medium_eval'] = eval_lib.EvaluationSuite(list(range(1_000)), 960)
_eval_suites['small_eval'] = eval_lib.EvaluationSuite(list(range(100)), 960)
_eval_suites['tiny_eval'] = eval_lib.EvaluationSuite(list(range(10)), 960)
_eval_suites['micro_eval'] = eval_lib.EvaluationSuite([0], 960)


def available_suites() -> List[str]:
  return list(_eval_suites.keys())


def get_eval_suite(name: str) -> eval_lib.EvaluationSuite:
  """Gets a named evaluation suite."""
  if name not in _eval_suites:
    raise ValueError(f'Unknown eval suite {name}')

  # Copy the seeds, rather than returning a mutable object.
  suite = _eval_suites[name]
  return eval_lib.EvaluationSuite(list(suite.seeds), suite.max_episode_length)
