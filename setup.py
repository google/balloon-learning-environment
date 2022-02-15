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

"""Setup file for installing the BLE."""
import os
import pathlib
import setuptools
from setuptools.command import build_py
from setuptools.command import develop

current_directory = pathlib.Path(__file__).parent
description = (current_directory / 'README.md').read_text()

core_requirements = [
    'absl-py',
    'flax',
    'gin-config',
    'gym',
    'jax >= 0.2.28',
    'jaxlib >= 0.1.76',
    'opensimplex <= 0.3.0',
    's2sphere',
    'scikit-learn',
    'tensorflow',
    'tensorflow-probability',
    'transitions',
]

dopamine_requirements = [
    'dopamine-rl >= 4.0.0',
]

acme_requirements = [
    'dm-acme',
    'dm-haiku',
    'dm-reverb',
    'dm-sonnet',
    'rlax',
    'xmanager',
]


def generate_requirements_file(path=None):
  """Generates requirements.txt file needed for running Acme.

  It is used by Launchpad GCP runtime to generate Acme requirements to be
  installed inside the docker image. Acme itself is not installed from pypi,
  but instead sources are copied over to reflect any local changes made to
  the codebase.
  Args:
    path: path to the requirements.txt file to generate.
  """
  if not path:
    path = os.path.join(os.path.dirname(__file__), 'acme_requirements.txt')
  with open(path, 'w') as f:
    for package in set(core_requirements + dopamine_requirements +
                       acme_requirements):
      f.write(f'{package}\n')


class BuildPy(build_py.build_py):

  def run(self):
    generate_requirements_file()
    build_py.build_py.run(self)


class Develop(develop.develop):

  def run(self):
    generate_requirements_file()
    develop.develop.run(self)

cmdclass = {
    'build_py': BuildPy,
    'develop': Develop,
}

entry_points = {
    'gym.envs': [
        '__root__=balloon_learning_environment.env.balloon_env:register_env'
    ]
}


setuptools.setup(
    name='balloon_learning_environment',
    long_description=description,
    long_description_content_type='text/markdown',
    version='0.1.2',
    cmdclass=cmdclass,
    packages=setuptools.find_packages(),
    install_requires=core_requirements,
    extras_require={
        'dopamine': dopamine_requirements,
        'acme': acme_requirements,
    },
    package_data={
        '': ['*.msgpack', '*.pb', '*.gin'],
    },
    entry_points=entry_points,
    python_requires='>=3.7',
)
