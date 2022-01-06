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

r"""Combines sharded eval results into a single json file.

"""

import glob
import json
import os
from typing import Sequence

from absl import app
from absl import flags


# TODO(joshgreaves): Rename models to agents (including README).
flags.DEFINE_string('path', None, 'The path containing the shard results.')
flags.DEFINE_multi_string('models', None,
                          'The names of the methods in the directory.')
flags.DEFINE_boolean('pretty_json', False,
                     'If true, it will write json files with an indent of 2.')
flags.DEFINE_boolean('flight_paths', False,
                     'If True, will include flight paths.')
flags.mark_flags_as_required(['path', 'models'])
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  del argv  # Unused.


  for model in FLAGS.models:
    data = list()

    for path in glob.glob(os.path.join(FLAGS.path, f'{model}_*.json')):
      with open(path, 'r') as f:
        data.extend(json.load(f))

    data = sorted(data, key=lambda x: x['seed'])

    if not FLAGS.flight_paths:
      for d in data:
        d['flight_path'] = []

    with open(os.path.join(FLAGS.path, f'{model}.json'), 'w') as f:
      json.dump(data, f, indent=2 if FLAGS.pretty_json else None)


if __name__ == '__main__':
  app.run(main)
