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
import typing
from urllib import parse

from pkg_resources import parse_version

MODEL_SUBFOLDER = 'legion_model'
LEGION_PROJECT_DESCRIPTION = 'legion.project.yaml'
BASE_IMAGE = os.getenv('BASE_IMAGE', 'legion/mlflow-toolchain:latest')
ENTRYPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'entrypoint.py')

try:
    import mlflow
    import mlflow.tracking
    import mlflow.tracking.utils
    import mlflow.projects
    import mlflow.models
    import mlflow.pyfunc
    import mlflow.store.artifact_repository_registry
except ImportError as import_error:
    print(f'Cannot import MLflow module: {import_error}')
    sys.exit(1)

try:
    import yaml
except ImportError as import_error:
    print(f'Cannot import yaml dependent module: {import_error}')
    sys.exit(1)


class ModelTraining(typing.NamedTuple):
    """
    Declaration of model training entity
    """

    name: str
    version: str
    work_dir: str
    entrypoint: str
    hyper_parameters: typing.Dict[str, typing.Any]


def parse_model_training_entity(source_file: str) -> ModelTraining:
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

    spec = mt.get('spec')
    if not isinstance(spec, dict):
        raise ValueError(f'Cannot find spec field or it is not a dict in file {source_file}')

    name = spec.get('name')
    if not isinstance(name, str):
        raise ValueError(f'Name should be a string: {name!r} in spec: {spec}')

    version = spec.get('version')
    if not isinstance(version, str):
        raise ValueError(f'Version should be a string: {version!r} in spec: {spec}')

    # Prepare run arguments
    work_dir = spec.get('workDir')
    if not isinstance(work_dir, str):
        raise ValueError(f'Incorrect workDir: {work_dir} in spec: {spec}')

    entry_point = spec.get('entrypoint', 'main')
    if not isinstance(entry_point, str):
        raise ValueError(f'Entry point should be a string: {entry_point} in spec: {spec}')

    hyper_parameters = spec.get('hyperparameters', {})
    if not isinstance(hyper_parameters, dict):
        raise ValueError(f'Invalid hyperparameters: {hyper_parameters} in spec: {spec}')

    return ModelTraining(
        name=name,
        version=version,
        work_dir=work_dir,
        entrypoint=entry_point,
        hyper_parameters=hyper_parameters,
    )


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


def save_models(mlflow_run: mlflow.projects.SubmittedRun, model_training: ModelTraining, target_directory: str) -> None:
    """
    Save models after run
    """
    # Using internal API for getting store and artifacts location
    store = mlflow.tracking._get_store()
    artifact_uri = store.get_run(mlflow_run.run_id).info.artifact_uri
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
    mlflow_target_directory = os.path.join(target_directory, MODEL_SUBFOLDER)

    logging.info(f"Copying MLflow model from {model_source_folder} to {mlflow_target_directory}")

    logging.info('Preparing target directory')
    if not os.path.exists(mlflow_target_directory):
        os.makedirs(mlflow_target_directory)

    copytree(model_source_folder, mlflow_target_directory)

    model_obj = mlflow.models.Model.load(mlflow_target_directory)
    py_flavor = model_obj.flavors[mlflow.pyfunc.FLAVOR_NAME]

    env = py_flavor.get('env')
    if env:
        dependencies = 'conda'
        conda_path = os.path.join(MODEL_SUBFOLDER, env)
        logging.info(f'Conda env located in {conda_path}')
    else:
        raise Exception('Unknown type of env - empty')

    entrypoint_target = os.path.join(mlflow_target_directory, 'entrypoint.py')
    shutil.copyfile(ENTRYPOINT, entrypoint_target)

    project_file_path = os.path.join(target_directory, LEGION_PROJECT_DESCRIPTION)
    with open(project_file_path, 'w') as proj_stream:
        data = {
            'binaries': {
                'type': 'python',
                'dependencies': dependencies,
                'conda_path': conda_path
            },
            'model': {
                'name': model_training.name,
                'version': model_training.version,
                'workDir': MODEL_SUBFOLDER,
                'entrypoint': 'entrypoint'
            },
            'toolchain': {
                'name': 'mlflow',
                'version': mlflow.__version__
            },
            'legionVersion': '1.0'
        }

        yaml.dump(data, proj_stream)


def train_models(model_training: ModelTraining) -> mlflow.projects.SubmittedRun:
    """
    Start MLfLow run
    """
    logging.debug('Validating MLflow version')
    mlflow_version = parse_version(mlflow.__version__)
    if mlflow_version < parse_version('1.0') or mlflow_version > parse_version('1.0'):
        raise Exception(f'Unsupported version {mlflow_version}. Please use MLflow version 1.0.*')

    logging.info('Getting of tracking URI')
    tracking_uri = mlflow.tracking.utils.get_tracking_uri()
    if not tracking_uri:
        raise ValueError('Can not get tracking URL')
    logging.info(f"Using MLflow client placed at {tracking_uri}")

    logging.info('Creating MLflow client, setting tracking URI')
    mlflow.tracking.utils.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Registering of experiment on tracking server if it is not exist
    logging.info(f"Searching for experiment with name {model_training.name}")
    experiment = client.get_experiment_by_name(model_training.name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"Experiment {experiment_id} has been found")
    else:
        logging.info(f"Creating new experiment with name {model_training.name}")
        experiment_id = client.create_experiment(model_training.name)

        logging.info(f"Experiment {experiment_id} has been created")
        client.get_experiment_by_name(model_training.name)

    # Starting run and awaiting of finish of run
    logging.info(f"Starting MLflow's run function. Parameters: [project directory: {model_training.work_dir}, "
                 f"entry point: {model_training.entrypoint}, hyper parameters: {model_training.hyper_parameters},"
                 f"experiment id={experiment_id}]")
    run = mlflow.projects.run(
        uri=model_training.work_dir,
        entry_point=model_training.entrypoint,
        parameters=model_training.hyper_parameters,
        experiment_id=experiment_id,
        backend='local',
        synchronous=True
    )

    logging.info(f"MLflow's run function finished. Run ID: {run.run_id}")

    return run


def setup_logging(args: argparse.Namespace) -> None:
    """
    Setup logging instance
    """
    log_level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(format='[legion][%(levelname)5s] %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        level=log_level)


if __name__ == '__main__':
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
        mlflow_run = train_models(model_training)

        # Save MLflow models as Legion artifact
        save_models(mlflow_run, model_training, args.target)
    except Exception as e:
        error_message = f'Exception occurs during model training. Message: {e}'

        if args.verbose:
            logging.exception(error_message)
        else:
            logging.error(error_message)

        sys.exit(2)
