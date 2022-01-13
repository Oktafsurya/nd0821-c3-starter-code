import os
import pandas as pd
import pytest

FP_CWD = os.getcwd()
CLEANED_DATA = 'data/census_cleaned.csv'

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
    df = pd.read_csv(os.path.join(FP_CWD, CLEANED_DATA))
    return df

def test_cat_features(data, cat_features):
    """ Checks the categorical features are contain in our data column. """
    assert set(cat_features).issubset(set(data.columns))

def test_data_column_name_cleaned(data):
    """ Check that there are no spaces in the column names """
    col_names = data.columns
    for col in col_names:
        assert " " not in col, f"Found space character in feature {col}"


