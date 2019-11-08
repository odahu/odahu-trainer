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
import os
import re

from setuptools import setup, find_packages

PACKAGE_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(PACKAGE_ROOT_PATH, 'mlflowrunner', '__version__.py')


def extract_version() -> str:
    """
    Extract version from .py file using regex

    :return: legion version
    """
    with open(VERSION_FILE, 'rt') as version_file:
        file_content = version_file.read()
        VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
        mo = re.search(VSRE, file_content, re.M)
        if mo:
            return mo.group(1)
        else:
            raise RuntimeError("Unable to find version string in %s." % (file_content,))


setup(
    name='legion-mlflow-runner',
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
        # TODO: change to PyPi when we publish release
        'legion-sdk @ git+https://github.com/legion-platform/legion.git@1.0.0-rc18#egg=legion-sdk&subdirectory=legion/sdk',
        'mlflow<=1.3.0,>=1.0.0',
        'PyYAML>=3.1.2'
    ],
    version=extract_version()
)
