import numpy as np
import pandas as pd
from odahuflow.trainer.helpers.templates.entrypoint import _extract_df_properties


def test_extract_df_properties():
    df = pd.DataFrame(
        {
            'A': 1.,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
            'D': np.array([3] * 4, dtype='int32'),
            'F': 'foo'
        }
    )

    # For now, we assume that the order of columns will be the same as in the input DataFrame
    assert _extract_df_properties(df) == [
        {'example': 0, 'name': 'A', 'required': True, 'type': 'number'},
        {'example': None, 'name': 'B', 'required': True, 'type': None},
        {'example': 0, 'name': 'C', 'required': True, 'type': 'number'},
        {'example': 0, 'name': 'D', 'required': True, 'type': 'integer'},
        {'example': '', 'name': 'F', 'required': True, 'type': 'string'}
    ]


def test_extract_empty_df_properties():
    assert _extract_df_properties(pd.DataFrame({})) == []
