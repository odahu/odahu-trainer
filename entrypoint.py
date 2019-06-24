import os
import json
import time
import functools
import sys

import numpy as np
import pandas as pd

import mlflow.models
import mlflow.pyfunc


MODEL_FLAVOR = None
MODEL_NAME = os.getenv('MODEL_NAME', 'model')
MODEL_LOCATION = os.getenv('MODEL_LOCATION', 'model')
MODEL_DATA_HEAD_FILE = os.path.join(MODEL_LOCATION, 'head.pkl')


class Timer:
    def __init__(self, name='UNKNOWN'):
        self.name = name

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print('{} took {}'.format(self.name, self.interval))


class NumpyEncoder(json.JSONEncoder):
    ''' Special json encoder for numpy types.
    Note that some numpy types doesn't have native python equivalence,
    hence json.dumps will raise TypeError.
    In this case, you'll need to convert your numpy types into its closest python equivalence.
    '''
    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, np.generic):
            return np.asscalar(o)
        return json.JSONEncoder.default(self, o)


def _get_jsonable_obj(data, pandas_orient='records'):
    '''Attempt to make the data json-able via standard library.
    Look for some commonly used types that are not jsonable and convert them into json-able ones.
    Unknown data types are returned as is.

    :param data: data to be converted, works with pandas and numpy, rest will be returned as is.
    :param pandas_orient: If `data` is a Pandas DataFrame, it will be converted to a JSON
                          dictionary using this Pandas serialization orientation.
    '''
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient=pandas_orient)
    if isinstance(data, pd.Series):
        return pd.DataFrame(data).to_dict(orient=pandas_orient)
    else:  # by default just return whatever this is and hope for the best
        return data


# def parse_json_input(json_input, orient='split'):
#     '''
#     :param json_input: A JSON-formatted string representation of a Pandas DataFrame, or a stream
#                        containing such a string representation.
#     :param orient: The Pandas DataFrame orientation of the JSON input. This is either 'split'
#                    or 'records'.
#     '''
#     return pd.DataFrame(data=json_input['data'], columns=json_input['columns'])


def init():
    model = mlflow.models.Model.load(MODEL_LOCATION)
    if mlflow.pyfunc.FLAVOR_NAME not in model.flavors:
        raise ValueError('{} not in model\'s flavors'.format(mlflow.pyfunc.FLAVOR_NAME))

    global MODEL_FLAVOR
    MODEL_FLAVOR = mlflow.pyfunc.load_model(MODEL_LOCATION)


def predict(input_object_or_dict):
    #with Timer('Parsing of JSON input to DataFrame (already de-serialized)'):
    #    df = pd.DataFrame({k: [v] for k, v in input_object_or_dict.items()})
    #    #df = parse_json_input(input_object_or_dict)

    columns = columns_order()
    if not columns:
        raise Exception('Columns order is unknown during inference phase')

    input_vector = [[input_object_or_dict[column] for column in columns]]
    #input_vector = [list(input_object_or_dict.values())]

    with Timer('Executing prediction on parsed DataFrame'):
        #result = MODEL_FLAVOR.predict(df)
        result = MODEL_FLAVOR.predict(input_vector)
    return _get_jsonable_obj(result[0])


def predict_matrix(input_matrix):
    with Timer('Executing prediction on matrix'):
        result = MODEL_FLAVOR.predict([input_matrix])
    return _get_jsonable_obj(result[0])


@functools.lru_cache()
def columns_order():
    if os.path.exists(MODEL_DATA_HEAD_FILE):
        head = pd.read_pickle(MODEL_DATA_HEAD_FILE)
    else:
        print('Columns order can not be captured', file=sys.__stderr__)
        return None

    return tuple(head.columns)


@functools.lru_cache()
def info():
    if os.path.exists(MODEL_DATA_HEAD_FILE):
        head = pd.read_pickle(MODEL_DATA_HEAD_FILE)
        print(list(head.columns))
    else:
        print('Columns order can not be captured', file=sys.__stderr__)
        return None

    return {
        'title': 'PredictionParameters',
        'description': 'Parameters for prediction',
        'type': 'object',
        'properties': {
            column: {
                'title': column,
                'type': head.dtypes[pos]
            }
            for pos, column in enumerate(head.columns)
        },
        'required': list(head.columns),
    }


def get_json_encoder() -> type:
    return NumpyEncoder


if __name__ == '__main__':
    #print(repr(columns_order()))
    print(repr(info()))
    init()
    print(repr(predict_matrix(
        [7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8]
    )))

    print(repr(predict({
        'total sulfur dioxide': 170,
        'volatile acidity': 0.27,
        'fixed acidity': 7,
        'citric acid': 0.36,
        'residual sugar': 20.7,
        'chlorides': 0.045,
        'free sulfur dioxide': 45,
        'density': 1.001,
        'pH': 3,
        'sulphates': 0.45,
        'alcohol': 8.8
    })))

    # print(repr(predict(
    #     {
    #         'columns': [
    #             'alcohol',
    #             'chlorides',
    #             'citric acid',
    #             'density',
    #             'fixed acidity',
    #             'free sulfur dioxide',
    #             'pH',
    #             'residual sugar',
    #             'sulphates',
    #             'total sulfur dioxide',
    #             'volatile acidity'
    #         ],
    #         'data': [
    #             [8.8, 0.045, 0.36, 1.001, 7, 45, 3, 20.7, 0.45, 170, 0.27]
    #         ]
    #     }
    # )))
