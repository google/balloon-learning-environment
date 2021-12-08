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

"""Setup file for installing the BLE."""
import setuptools

setuptools.setup(
    name='balloon_learning_environment',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'absl-py',
        'dopamine-rl >= 4.0.0',
        'flax',
        'gcsfs',
        'gin-config',
        'gym',
        'opensimplex',
        's2sphere',
        'scikit-learn',
        'tensorflow',
        'tensorflow-datasets >= 4.4.0',
        'tensorflow-probability',
        'transitions',
        'zarr',
    ],
    package_data={
        '': ['*.msgpack', '*.pb', '*.gin'],
    },
    python_requires='>=3.7',
)
