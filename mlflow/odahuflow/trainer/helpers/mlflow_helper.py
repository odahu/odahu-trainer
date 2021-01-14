import json
import logging
import os
import shutil
from urllib import parse


import yaml

from odahuflow.sdk.gppi.executor import GPPITrainedModelBinary
from odahuflow.sdk.models import K8sTrainer, ModelTraining
from odahuflow.trainer.helpers.conda import run_mlflow_wrapper, update_model_conda_env
from odahuflow.trainer.helpers.fs import copytree
import mlflow
import mlflow.models
import mlflow.projects
import mlflow.pyfunc
import mlflow.tracking
from mlflow.tracking import MlflowClient, get_tracking_uri, set_tracking_uri

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
                raise ValueError(f'Cannot decode ModelTraining resource file: {decode_error}') from decode_error

    return K8sTrainer.from_dict(mt)


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
        raise Exception('Can not find any model')

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

    project_file_path = os.path.join(target_directory, ODAHUFLOW_PROJECT_DESCRIPTION)
    with open(project_file_path, 'w') as proj_stream:
        data = {
            'binaries': {
                'type': 'python',
                'dependencies': dependencies,
                'conda_path': conda_path
            },
            'model': {
                'name': model_training.spec.model.name,
                'version': model_training.spec.model.version,
                'workDir': MODEL_SUBFOLDER,
                'entrypoint': 'entrypoint'
            },
            'toolchain': {
                'name': 'mlflow',
                'version': mlflow.__version__
            },
            'odahuflowVersion': '1.0',
            'output': {
                'run_id': mlflow_run_id
            }
        }

        yaml.dump(data, proj_stream)

    logging.info("GPPI stored. Start to GPPI validation")
    mb = GPPITrainedModelBinary(target_directory)
    mb.self_check()
    logging.info("GPPI is validated. OK")


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
