from lib.ml.models import *

import pandas as pd

DATASET = "data/380000_final.csv"

# Load the data
df = pd.read_csv(DATASET)
df.dropna(axis=0, inplace=True)
# df.drop(df[(df.lang != "en")].index, inplace=True)
df = df[["genre", "lyrics"]]
df.drop(df[(df.genre == "Not Available") | (df.genre == "Other")].index, inplace=True)

# Build model and perform predictions
log_model = Model("FeatureSelection_RF", "tfidf",
                  "Gradient Boosting - Advanced Cleaning, N/A, Other Removed, TF-IDF (380)",
                  "shallow/43_gb_advanced_tfidf_380")
log_model.train_predict(df)
