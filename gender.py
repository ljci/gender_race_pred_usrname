import numpy as np
import pandas as pd
import re
import os
import warnings
import tensorflow as tf
import joblib
from multiprocessing import Pool


# Define a function to predict gender
def predict_gender(name):
    if isinstance(name, str):  # Check if name is a string
        name = [name.lower()]  # Prepare the name for prediction
        predicted_gender = loaded_model.predict(
            name
        )  # Use the loaded model for prediction
        return (
            "Male" if predicted_gender[0] == 0 else "Female"
        )  # Return the predicted gender
    return "Unknown"  # Return a default value for non-string names


def process_row(name):
    return predict_gender(name)


if __name__ == "__main__":
    warnings = warnings.filterwarnings("ignore")
    df = pd.read_csv("./race_gender_data/df.csv").iloc[:, 1:]
    loaded_model = joblib.load("./gender/gender_predictor.pkl")
    print(df.shape)
    # Set up a process pool
    num_processes = os.cpu_count()

    # Parallelize predictions with Pool
    with Pool(num_processes) as pool:
        df["pred_gender"] = pool.map(process_row, df["fullname"])

    # Save the results
    print(df.shape)
    df = df.dropna()
    df = df.drop_duplicates()
    df.to_csv("./race_gender_data/df_gender.csv", index=False)
