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
import json
import os
import shutil
from os.path import join
from typing import Any, Dict

import yaml
from odahuflow.mlflowrunner.wrapper.entities import MLFlowWrapperOutput
from odahuflow.sdk import io_proc_utils
from odahuflow.sdk.models import ModelTraining

MLFLOW_WRAPPER_OUTPUT_FILE_PATH = 'mlflow-output.json'
MLFLOW_WRAPPER_INPUT_FILE_PATH = "mlflow-input.json"
MLPROJECT_FILE_NAME = "mlproject"
DEFAULT_CONDA_FILE_NAME = "conda.yaml"
ODAHU_MODEL_CONDA_ENV_NAME = os.environ.get("ODAHU_CONDA_ENV_NAME", "odahu_model")


def _find_mlproject_file_path(model_training: ModelTraining) -> str:
    work_dir = join(os.getcwd(), model_training.spec.work_dir)

    filenames = os.listdir(work_dir)
    for filename in filenames:
        if filename.lower() == MLPROJECT_FILE_NAME:
            return join(work_dir, filename)

    raise ValueError(f"Can't find a conda dependencies file in the '{work_dir}' dir")


def _extract_conda_file_name(mlproject_file_path: str) -> str:
    """
    Extract conda dependencies file name from MLProject file
    :param mlproject_file_path: MLFlow MLProject file path
    :return: conda file name
    """
    with open(mlproject_file_path) as f:
        ml_project = yaml.load(f)

        return ml_project.get("conda_env", DEFAULT_CONDA_FILE_NAME)


def update_model_conda_env(model_training: ModelTraining):
    """
    Update model conda dependencies
    :param model_training:
    """
    io_proc_utils.run(
        "conda", "env", "update", "-n", ODAHU_MODEL_CONDA_ENV_NAME,
        "-f", _extract_conda_file_name(_find_mlproject_file_path(model_training)),
        cwd=os.path.join(os.getcwd(), model_training.spec.work_dir)
    )


def run_mlflow_wrapper(mlflow_input: Dict[str, Any]) -> str:
    """
    Prepare parameters and run MLFlow wrapper inside the model conda environment
    :param mlflow_input: parameters which will be passed to mlflow.run function
    :return: MLFlow run ID
    """
    with open(MLFLOW_WRAPPER_INPUT_FILE_PATH, 'w') as f:
        json.dump(mlflow_input, f)

    io_proc_utils.run(
        "conda", "run", "-n", ODAHU_MODEL_CONDA_ENV_NAME,
        shutil.which('odahu-flow-mlflow-wrapper'),
        "--input", MLFLOW_WRAPPER_INPUT_FILE_PATH,
        "--output", MLFLOW_WRAPPER_OUTPUT_FILE_PATH,
    )

    with open(MLFLOW_WRAPPER_OUTPUT_FILE_PATH) as f:
        return MLFlowWrapperOutput(**json.load(f)).run_id
