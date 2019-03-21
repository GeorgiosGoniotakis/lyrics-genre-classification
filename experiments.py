from lib.utils.preprocess import *
from lib.ml.models import *

import pandas as pd

DATASET = "data/380000_clean.csv"
BALANCED_DATASET = "data/380000_balanced.csv"

# Load the data
df = pd.read_csv(BALANCED_DATASET)
df.dropna(axis=0, inplace=True)


# TODO: Preprocess the data set
df = preprocess_data(df, col="lyrics")
df.to_csv("data/380000_balanced_clean.csv", index=False)

# Build model and perform predictions
log_model = Model("LogisticRegression", "tfidf", "Logistic Regression - Balanced, Basic Cleaning (380)",
                  "2_logreg_balanced_basic_cleaning_380")
log_model.train_predict(df)
