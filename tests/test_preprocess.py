# tests/test_preprocess.py

import pandas as pd
from src.preprocess import prepare_data

def test_prepare_data_shapes():
    """
    Ensure prepare_data returns X and y with matching lengths and correct dimensions.
    """
    X, y = prepare_data()
    # X should be a DataFrame with one column
    assert hasattr(X, 'shape')
    assert X.shape[1] == 1
    # y should have the same number of rows as X
    assert len(X) == len(y)
