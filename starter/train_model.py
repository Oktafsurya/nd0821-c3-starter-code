# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('../data/census_cleaned.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
trained_model = train_model(X_train, y_train)
pd.to_pickle(trained_model, "starter/model/rf_model.pkl")

# Train and save a model.
pd.to_pickle(encoder, '../model/ohe.pkl')
pd.to_pickle(lb, '../model/lb.pkl')
