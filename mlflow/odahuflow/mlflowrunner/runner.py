#!/usr/bin/env python3
#
#    Copyright 2019 EPAM Systems
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
import argparse
import json
import logging
import os
import os.path
import shutil
import sys
import tempfile
from pathlib import Path
from urllib import parse

import mlflow
import mlflow.models
import mlflow.projects
import mlflow.pyfunc
import mlflow.tracking
import yaml
from mlflow.tracking import set_tracking_uri, get_tracking_uri, MlflowClient
from odahuflow.gppi.model.dependencies import Dependencies, CondaDependencies
from odahuflow.gppi.model.meta import Meta, ToolchainMeta, ModelMeta
from odahuflow.gppi.python.creator import PythonModelCreator
from odahuflow.mlflowrunner.conda import update_model_conda_env, run_mlflow_wrapper
from odahuflow.mlflowrunner.library import ModelLibraryInfo, ModelLibraryPackage, ModelLibraryEntrypoint, \
    generate_model_library
from odahuflow.sdk.models import K8sTrainer
from odahuflow.sdk.models import ModelTraining

MODEL_SUBFOLDER = 'odahuflow_model'
ODAHUFLOW_PROJECT_DESCRIPTION = 'odahuflow.project.yaml'
ENTRYPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'entrypoint.py')


def parse_model_training_entity(source_file: str) -> K8sTrainer:
    """
    Parse model training file
    """
    logging.info(f'Parsing Model Training file: {source_file}')

    # Validate resource file exist
    if not os.path.exists(source_file) or not os.path.isfile(source_file):
        raise ValueError(f'File {source_file} is not readable')

    with open(source_file, 'r') as mt_file:
        mt = mt_file.read()
        logging.debug(f'Content of {source_file}:\n{mt}')

        try:
            mt = json.loads(mt)
        except json.JSONDecodeError:
            try:
                mt = yaml.safe_load(mt)
            except json.JSONDecodeError as decode_error:
                raise ValueError(f'Cannot decode ModelTraining resource file: {decode_error}')

    return K8sTrainer.from_dict(mt)


def copytree(src, dst):
    """
    Copy file tree from <src> location to <dst> location
    """
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def save_models(mlflow_run_id: str, model_training: ModelTraining, target_directory: str) -> None:
    """
    Save models after run
    """
    # Using internal API for getting store and artifacts location
    store = mlflow.tracking._get_store()
    artifact_uri = store.get_run(mlflow_run_id).info.artifact_uri
    logging.info(f"Artifacts location detected. Using store {store}")

    parsed_url = parse.urlparse(artifact_uri)
    if parsed_url.scheme and parsed_url.scheme != 'file':
        raise ValueError(f'Unsupported scheme of url: {parsed_url}')

    artifact_uri = parsed_url.path

    logging.info(f"Analyzing directory {artifact_uri} for models")
    models = []
    for subpath in os.listdir(artifact_uri):
        full_subpath = os.path.join(artifact_uri, subpath)
        ml_model_location = os.path.join(full_subpath, 'MLmodel')

        if os.path.isdir(full_subpath) and os.path.exists(ml_model_location):
            logging.debug(f"Analyzing {subpath} in {full_subpath}")

            try:
                model = mlflow.models.Model.load(full_subpath)

                flavors = model.flavors.keys()
                logging.debug(f"{subpath} contains {flavors} flavours")
                if mlflow.pyfunc.FLAVOR_NAME not in flavors:
                    logging.debug(f"{flavors} does not has {mlflow.pyfunc.FLAVOR_NAME} flavor, skipping")
                    continue

                logging.info(f"Registering model {subpath}")
                models.append(subpath)
            except Exception as load_exception:
                logging.debug(f"{full_subpath} is not a MLflow model: {load_exception}")

    if len(models) > 1:
        raise Exception(f'Founded models: {models!r}. Only 1 model allowed')

    if not models:
        raise Exception(f'Can not find any model')

    model = models[0]
    model_source_folder = os.path.join(artifact_uri, model)

    model_obj = mlflow.models.Model.load(model_source_folder)
    py_flavor = model_obj.flavors[mlflow.pyfunc.FLAVOR_NAME]

    env = py_flavor.get('env')
    if not env:
        raise Exception('Unknown type of env - empty')

    conda_path = os.path.join(MODEL_SUBFOLDER, env)
    logging.info(f'Conda env located in {conda_path}')

    meta = Meta(
        toolchain=ToolchainMeta(
            name='mlflow',
            version=mlflow.__version__
        ),
        model=ModelMeta(
            name=model_training.spec.model.name,
            version=model_training.spec.model.version,
        ),
        output={
            'run_id': mlflow_run_id
        },
        dependencies=Dependencies(
            conda=CondaDependencies(
                source_file_name=conda_path,
            )
        )
    )
    info = ModelLibraryInfo(
        model_entrypoint_name="model-1-2-3",
        entrypoint=ModelLibraryEntrypoint(dir="model"),
        package=ModelLibraryPackage(),
        binaries_path=Path(model_source_folder),
    )

    meta.dump_to_file(Path(model_source_folder))

    with tempfile.TemporaryDirectory() as temp_dir_library_path:
        temp_library_path = Path(temp_dir_library_path) / 'library.tar.gz'
        generate_model_library(info, temp_library_path)

        PythonModelCreator(
            meta=meta,
            output_artifact=Path(target_directory) / 'model.tar.gz',
            library=temp_library_path,
        ).create()


def train_models(model_training: ModelTraining) -> str:
    """
    Start MLfLow run
    """
    logging.info('Downloading conda dependencies')
    update_model_conda_env(model_training)

    logging.info('Getting of tracking URI')
    tracking_uri = get_tracking_uri()
    if not tracking_uri:
        raise ValueError('Can not get tracking URL')
    logging.info(f"Using MLflow client placed at {tracking_uri}")

    logging.info('Creating MLflow client, setting tracking URI')
    set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Registering of experiment on tracking server if it is not exist
    logging.info(f"Searching for experiment with name {model_training.spec.model.name}")
    experiment = client.get_experiment_by_name(model_training.spec.model.name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"Experiment {experiment_id} has been found")
    else:
        logging.info(f"Creating new experiment with name {model_training.spec.model.name}")
        experiment_id = client.create_experiment(model_training.spec.model.name)

        logging.info(f"Experiment {experiment_id} has been created")
        client.get_experiment_by_name(model_training.spec.model.name)

    # Starting run and awaiting of finish of run
    logging.info(f"Starting MLflow's run function. Parameters: [project directory: {model_training.spec.work_dir}, "
                 f"entry point: {model_training.spec.entrypoint}, "
                 f"hyper parameters: {model_training.spec.hyper_parameters}, "
                 f"experiment id={experiment_id}]")

    mlflow_input = {
        "uri": model_training.spec.work_dir,
        "entry_point": model_training.spec.entrypoint,
        "parameters": model_training.spec.hyper_parameters,
        "experiment_id": experiment_id,
        "backend": 'local',
        "synchronous": True,
        "use_conda": False,
    }

    run_id = run_mlflow_wrapper(mlflow_input)

    # TODO: refactor
    client.set_tag(run_id, "training_id", model_training.id)
    client.set_tag(run_id, "model_name", model_training.spec.model.name)
    client.set_tag(run_id, "model_version", model_training.spec.model.version)

    logging.info(f"MLflow's run function finished. Run ID: {run_id}")

    return run_id


def setup_logging(args: argparse.Namespace) -> None:
    """
    Setup logging instance
    """
    log_level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(format='[odahuflow][%(levelname)5s] %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        level=log_level)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="more extensive logging")
    parser.add_argument("--mt-file", '--mt', type=str, required=True,
                        help="json/yaml file with a mode training resource")
    parser.add_argument("--target", type=str, default='mlflow_output',
                        help="directory where result model will be saved")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args)
    try:
        # Parse ModelTraining entity
        model_training = parse_model_training_entity(args.mt_file)

        # Start MLflow training process
        mlflow_run_id = train_models(model_training.model_training)

        # Save MLflow models as odahuflow artifact
        save_models(mlflow_run_id, model_training.model_training, args.target)
    except Exception as e:
        error_message = f'Exception occurs during model training. Message: {e}'

        if args.verbose:
            logging.exception(error_message)
        else:
            logging.error(error_message)

        sys.exit(2)


if __name__ == '__main__':
    main()
