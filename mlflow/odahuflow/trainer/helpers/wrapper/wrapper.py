#!/usr/bin/env python3
#
#    Copyright 2020 EPAM Systems
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
"""
This entrypoint starts MLFlow training in model conda environment.
Because the wrapper only must requires the Python STD library and MLFlow dependency.
"""
import argparse
import json
import logging
import sys
from typing import Any, Dict

from odahuflow.trainer.helpers.wrapper.entities import MLFlowWrapperOutput
from pkg_resources import parse_version
import mlflow
import mlflow.models
import mlflow.projects
import mlflow.pyfunc
import mlflow.tracking


def work(input_file_path: str, output_file_path: str):
    """
    Launch mlflow run process

    :param input_file_path: file with MLFlow input parameters
    :param output_file_path: file where MLFlow output will be stored
    """
    logging.debug('Validating MLflow version')
    mlflow_version = parse_version(mlflow.__version__)
    if mlflow_version < parse_version('1.0') or mlflow_version >= parse_version('2.0'):
        raise Exception(f'Unsupported version {mlflow_version}. Please use MLflow versions >= 1.* but =< 2.* ')

    logging.debug("Reading mlflow input parameters")
    with open(input_file_path) as f:
        mlflow_input: Dict[str, Any] = json.load(f)

    logging.debug('Validating MLflow version')
    run = mlflow.projects.run(
        **mlflow_input
    )

    with open(output_file_path, 'w') as f:
        json.dump(MLFlowWrapperOutput(run_id=run.run_id)._asdict(), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="json/yaml file with a mode training resource")
    parser.add_argument("--output", type=str, required=True,
                        help="json/yaml file with a mode training resource")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)

    try:
        work(args.input, args.output)
    except Exception:
        logging.exception('Exception occurs during model training')
        sys.exit(2)
