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
import sys
import os
import os.path
import re
import json
import typing
import tempfile
import shutil
from pkg_resources import parse_version

MLFLOW_SUBDIR = 'mlflow'
LEGION_PROJECT_DESCRIPTION = 'legion.project.yaml'
LEGION_MODEL_DESCRIPTION = 'legion.model.yaml'

try:
    import mlflow
    import mlflow.tracking
    import mlflow.tracking.utils
    import mlflow.projects
    import mlflow.models
    import mlflow.pyfunc
    import mlflow.store.artifact_repository_registry
except ImportError as import_error:
    print('Cannot import MLflow module: {}'.format(import_error))
    sys.exit(1)

try:
    import yaml
except ImportError as import_error:
    print('Cannot import MLflow dependent module: {}'.format(import_error))
    sys.exit(1)


ModelTraining = typing.NamedTuple('ModelTraining', (
    ('name', str),
    ('workDir', str),
    ('entrypoint', str),
    ('hyperparameters', typing.Dict[str, typing.Any]),
))

BASE_IMAGE = os.getenv('BASE_IMAGE', 'legion/mlflow-toolchain:latest')


def parse_model_training_entity(source_file):
    """
    Parse model training file
    """
    # Validate resource file exist
    if not os.path.exists(source_file) or not os.path.isfile(source_file):
        print('File {} is not readable'.format(source_file))
        return None

    # Load resource file
    with open(source_file, 'r') as description_stream:
        try:
            description = yaml.load(description_stream)
        except json.JSONDecodeError as decode_error:
            print('Cannot decode ModelTraining resource file: {}'.format(decode_error))
            return None

    metadata = description.get('metadata')
    if not isinstance(metadata, dict):
        print('Cannot find metadata field or it is not a dict')
        return None

    name = metadata.get('name')
    if not isinstance(name, str):
        print('Name should be a string: {!r}'.format(name))
        return None

    spec = description.get('spec')
    if not isinstance(spec, dict):
        print('Cannot find spec field or it is not a dict')
        return None

    # Prepare run arguments
    workDir = spec.get('workDir', None)
    if not isinstance(workDir, str):
        print('Incorrect workDir: {!r}'.format(workDir))
        return None

    entry_point = spec.get('entrypoint', 'main')
    if not isinstance(entry_point, str):
        print('Entry point should be a string: {!r}'.format(entry_point))
        return None

    parameters = spec.get('hyperparameters', {})
    if not isinstance(parameters, dict):
        print('Invalid hyperparameters: {!r}'.format(parameters))
        return None

    return ModelTraining(
        name=name,
        workDir=workDir,
        entrypoint=entry_point,
        hyperparameters=parameters
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


def main(source_file, target_directory):
    # Parsing training resource file
    model_training = parse_model_training_entity(source_file)  # type: typing.Optional[ModelTraining]
    if not model_training:
        print('Cannot parse input information')
        return False

    # Preparing target directory
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Validating MLFlow version
    mlflow_version = parse_version(mlflow.__version__)
    if mlflow_version < parse_version('1.0') or mlflow_version > parse_version('1.0'):
        print('Unsupported version {}. Please use MLFlow version 1.0.*'.format(mlflow_version))
        return False

    # Getting of tracking URI
    tracking_uri = mlflow.tracking.utils.get_tracking_uri()
    if not tracking_uri:
        print('Can not get tracking URL')
        return False

    # Creating MLFlow client, setting tracking URI
    print("=== Using MLFlow client placed at {!r} ===".format(tracking_uri))
    mlflow.tracking.utils.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Registering of experiment on tracking server if it is not exist
    print("=== Searching for experiment with name {!r} ===".format(model_training.name))
    experiment = client.get_experiment_by_name(model_training.name)

    if experiment:
        experiment_id = experiment.experiment_id
        print("=== Experiment {!r} has been found ===".format(experiment_id))
    else:
        print("=== Creating new experiment with name {!r} ===".format(model_training.name))
        experiment_id = client.create_experiment(model_training.name)
        print("=== Experiment {!r} has been created ===".format(experiment_id))
        experiment = client.get_experiment_by_name(model_training.name)

    # Starting run and awaiting of finish of run
    print("=== Starting MLFlow's run function ===")
    print("Project directory: {!r}".format(model_training.workDir))
    print("Entry point: {!r}".format(model_training.entrypoint))
    print("Parameters: {!r}".format(model_training.hyperparameters))
    print("Experiment ID: {!r}".format(experiment_id))
    run = mlflow.projects.run(
        uri=model_training.workDir,
        entry_point=model_training.entrypoint,
        parameters=model_training.hyperparameters,
        experiment_id=experiment_id,
        backend='local',
        synchronous=True
    )
    print("=== MLflow's run function finished ===")
    print("Run ID: {!r}".format(run.run_id))

    # Using internal API for getting store and artifacts location
    store = mlflow.tracking._get_store()
    artifact_uri = store.get_run(run.run_id).info.artifact_uri
    print("=== Using store {!r} ===".format(store))
    print("=== Artifacts location detected ===")
    print(artifact_uri)

    print("=== Analyzing directory {!r} for models ===".format(artifact_uri))
    models = []
    for subpath in os.listdir(artifact_uri):
        full_subpath = os.path.join(artifact_uri, subpath)
        ml_model_location = os.path.join(full_subpath, 'MLmodel')
        if os.path.isdir(full_subpath) and os.path.exists(ml_model_location):
            print("=== Analyzing {} ({!r}) ===".format(subpath, full_subpath))
            try:
                model = mlflow.models.Model.load(full_subpath)
            except Exception as load_exception:
                print("=== {!r} is not a MLFlow model: {} ===".format(full_subpath, load_exception))
            flavors = model.flavors.keys()
            print("=== {} contains {} flavours ===".format(subpath, tuple(flavors)))
            if mlflow.pyfunc.FLAVOR_NAME not in flavors:
                print("=== {} does not has {} flavor, skipping ===".format(mlflow.pyfunc.FLAVOR_NAME))
                continue
            print("=== Registering model {} ===".format(subpath))
            models.append(subpath)

    mlflow_target_directory = os.path.join(target_directory, MLFLOW_SUBDIR)
    print("=== Copying MLFlow models from {!r} to {!r} ===".format(artifact_uri, mlflow_target_directory))
    if not os.path.exists(mlflow_target_directory):
        os.makedirs(mlflow_target_directory)
    copytree(artifact_uri, mlflow_target_directory)

    mlflow_models_list = os.path.join(target_directory, LEGION_PROJECT_DESCRIPTION)
    print("=== Dumping MLFlow models list to {!r} ===".format(mlflow_models_list))
    with open(mlflow_models_list, 'w') as proj_stream:
        yaml.dump({
            'models': [{
                'name': model,
                'location': os.path.join(MLFLOW_SUBDIR, model)
            } for model in models]
        }, proj_stream)

    for model in models:
        location = os.path.join(mlflow_target_directory, model)
        print("=== Processing model {!r} in location {!r} ===".format(model, location))
        model = mlflow.models.Model.load(location)
        py_flavor = model.flavors[mlflow.pyfunc.FLAVOR_NAME]
        dependencies = None
        conda_path = None
        env = py_flavor.get('env')
        if env:
            dependencies = 'conda'
            conda_path = env
        else:
            raise Exception('Unknown type of env - empty')

        data = {
            'binaries': {
                'type': 'python',
                'dependencies': dependencies,
                'conda_path': conda_path
            },
            'toolchain': {
                'name': 'mlflow',
                'version': mlflow.__version__
            },
            'legionVersion': '1.0'
        }

        with open(os.path.join(location, LEGION_MODEL_DESCRIPTION), 'w') as model_descr_stream:
            yaml.dump(data, model_descr_stream)

    return True


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('This script should be run with two arguments - path to ModelTraining resource and path to save final resources to',
              file=sys.stderr)
        sys.exit(1)

    if not main(sys.argv[1], sys.argv[2]):
        print('Failed to start run. Interrupting', file=sys.stderr)
        sys.exit(2)

