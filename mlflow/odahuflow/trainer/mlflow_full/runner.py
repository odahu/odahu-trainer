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
import sys

from odahuflow.trainer.helpers.log import setup_logging
from odahuflow.trainer.helpers.mlflow_helper import parse_model_training_entity, train_models, save_models


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
