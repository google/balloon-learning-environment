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

import dataclasses
from typing import List, Sequence


@dataclasses.dataclass
class EvaluationSuite:
  """An evaluation suite specification.

  Attributes:
    seeds: A sequence of seeds to evaluate the agent on.
    max_episode_length: The maximum number of steps to evaluate the agent
      on one seed. Must be greater than 0.
  """
  seeds: Sequence[int]
  max_episode_length: int


_eval_suites = dict()


_eval_suites['big_eval'] = EvaluationSuite(list(range(10_000)), 960)
_eval_suites['medium_eval'] = EvaluationSuite(list(range(1_000)), 960)
_eval_suites['small_eval'] = EvaluationSuite(list(range(100)), 960)
_eval_suites['tiny_eval'] = EvaluationSuite(list(range(10)), 960)
_eval_suites['micro_eval'] = EvaluationSuite([0], 960)


def available_suites() -> List[str]:
  return list(_eval_suites.keys())


def get_eval_suite(name: str) -> EvaluationSuite:
  """Gets a named evaluation suite."""
  if name not in _eval_suites:
    raise ValueError(f'Unknown eval suite {name}')

  # Copy the seeds, rather than returning a mutable object.
  suite = _eval_suites[name]
  return EvaluationSuite(list(suite.seeds), suite.max_episode_length)
