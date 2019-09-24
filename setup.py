#!/usr/bin/env python3
#
#    Copyright 2019 EPAM Systems
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

from setuptools import setup, find_packages

setup(
    name='legion_mlflow_runner',
    author='Legion Platform Team',
    license='Apache v2',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='mlflow legion',
    python_requires='>=3.6',
    packages=find_packages(),
    data_files=[('', ["README.md"])],
    zip_safe=False,
    entry_points={
        'console_scripts': ['legion-mlflow-runner=mlflowrunner.runner:main'],
    },
    install_requires=[
        # TODO: fix dependencies when we will publish the legion packages to pypi repo
        # TODO: consider Pipenv usage
        'legion-sdk @ git+https://github.com/legion-platform/legion.git@1.0.0-rc18#egg=legion-sdk&subdirectory=legion/sdk',
        'mlflow==1.0.0',
        'PyYAML>=3.1.2'
    ],
    version='1.0.0'
)
