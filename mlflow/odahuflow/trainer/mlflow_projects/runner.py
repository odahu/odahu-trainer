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
import argparse
import logging
import os
import shutil
import sys
import tempfile

import yaml
from odahuflow.sdk.models import ModelTraining
from odahuflow.trainer.helpers.log import setup_logging
from odahuflow.trainer.helpers.fs import copytree
from odahuflow.trainer.helpers.mlflow_helper import parse_model_training_entity, train_models

OUTPUT_DIR = "ODAHUFLOW_OUTPUT_DIR"
ODAHUFLOW_PROJECT_DESCRIPTION = "odahuflow.project.yaml"


def create_project_file(model_training: ModelTraining, project_file_path: str, mlflow_run_id: str):

    with open(project_file_path, 'w') as proj_stream:
        data = {
            'name': model_training.spec.model.name,
            'version': model_training.spec.model.version,
            'output': {
                'run_id': mlflow_run_id
            }
        }
        yaml.dump(data, proj_stream)


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

    output_dir = os.environ[OUTPUT_DIR] = tempfile.mkdtemp()
    logging.debug(f"output dir: {output_dir}")


    try:
        # Parse ModelTraining entity
        model_training = parse_model_training_entity(args.mt_file)

        # Start MLflow training process
        mlflow_run_id = train_models(model_training.model_training)

        # Copy $WORKDIR/data folder to output destination
        copytree(os.path.join(model_training.model_training.spec.work_dir, "data"), output_dir)

        # Create model name/version file
        project_file_path = os.path.join(output_dir, ODAHUFLOW_PROJECT_DESCRIPTION)
        create_project_file(model_training.model_training, project_file_path, mlflow_run_id)

        # copy output to target folder
        logging.info('Preparing target directory')
        if not os.path.exists(args.target):
            os.makedirs(args.target)
        copytree(output_dir, args.target)

        # rm temp directory
        shutil.rmtree(output_dir)

    except Exception as e:
        error_message = f'Exception occurs during model training. Message: {e}'

        if args.verbose:
            logging.exception(error_message)
        else:
            logging.error(error_message)

        sys.exit(2)


if __name__ == '__main__':
    main()
