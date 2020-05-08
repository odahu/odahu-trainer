#
#    Copyright 2020 EPAM Systems
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

from setuptools import find_packages, setup

requirements = ['odahu-flow-gppi']

setup(
    name='{{cookiecutter.package.name}}',
    version="{{cookiecutter.package.version}}",
    description='GPPI model',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'odahuflow.models': [
            '{{cookiecutter.model_entrypoint_name}} = {{cookiecutter.entrypoint.dir}}.{{cookiecutter.entrypoint.file}}:MLflowModel',
        ],
    },
)
