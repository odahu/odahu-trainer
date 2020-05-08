from os.path import dirname
from pathlib import Path
from typing import Any, Optional, List

import mlflow
import numpy as np
import pandas as pd
from mlflow import models, pyfunc
from odahuflow.gppi.model.meta import Meta
from odahuflow.gppi.model.model import ModelInputPredict, ModelOutputPredict, Model
from odahuflow.gppi.model.schema import ModelSchemas


class MLflowModel(Model):

    def __init__(self, binary_path: Optional[Path] = None):
        binary_path = binary_path or Path(dirname(__file__)) / 'binaries'

        self._model = _load_model(binary_path.joinpath('model.pkl'))
        self._input_sample = _load_sample(binary_path.joinpath('head_input.pkl'))
        self._output_sample = _load_sample(binary_path.joinpath('head_output.pkl'))
        schema = ModelSchemas.from_df(
            input=self._input_sample,
            output=self._output_sample
        )
        meta = Meta.read_from_file(binary_path)

        super().__init__(meta, schema)

    @property
    def raw_model(self) -> Any:
        return self._model

    def predict(
            self,
            input_matrix: ModelInputPredict,
            provided_columns_names: Optional[List[str]] = None,
    ) -> ModelOutputPredict:
        """
        Make prediction on a Matrix of values

        :param input_matrix: data for prediction
        :param provided_columns_names: Name of columns for provided matrix
        :return: result matrix and result column names
        """
        if provided_columns_names:
            input_matrix = pd.DataFrame(input_matrix, columns=provided_columns_names)
        else:
            input_matrix = pd.DataFrame(input_matrix)

        if provided_columns_names and self._input_sample is not None:
            input_matrix = input_matrix.reindex(columns=self._input_sample.columns)

        result = self.raw_model.predict(input_matrix)

        result_columns = []
        if self._output_sample is not None:
            result_columns = self._output_sample.columns

        # Register column names, overwrite if we've a sample
        if hasattr(result, 'columns'):
            result_columns = result.columns

        # TODO: think about better approach
        if isinstance(result, pd.DataFrame):
            output_matrix = result.to_numpy().tolist()
        elif isinstance(result, np.ndarray):
            output_matrix = result.tolist()
        else:
            output_matrix = result

        return output_matrix, tuple(result_columns)

    def verify(self):
        if self.info() is None:
            print('')

        if self._input_sample:
            self.predict(self._input_sample)


def _load_sample(schema_path: Path) -> Optional[pd.DataFrame]:
    if schema_path.exists():
        return pd.read_pickle(schema_path)

    return None


def _load_model(model_path: Path) -> Any:
    model = mlflow.models.Model.load(model_path)
    if pyfunc.FLAVOR_NAME not in model.flavors:
        raise ValueError('{} not in model\'s flavors'.format(pyfunc.FLAVOR_NAME))

    model = pyfunc.load_model(model_path)

    return model
