import os
import pandas as pd
import pytest
import sklearn
from starter.ml.data import process_data

CLEANED_DATA = '../data/census_cleaned.csv'

@pytest.fixture
def cat_features():
    cat_col = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_col

@pytest.fixture
def data():
    df = pd.read_csv(CLEANED_DATA)
    return df

def test_cat_features(data, cat_features):
    """ Checks the categorical features are contain in our data column. """
    assert set(cat_features).issubset(set(data.columns))

