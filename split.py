import numpy as np
import pandas as pd

if __name__ == "__main__":

    # read df_gender
    df = pd.read_csv("./race_gender_data/df_gender.csv")
    # split df_gender to df_single and df_others, clean both
    df_single = df[df["firstname"] == df["lastname"]].reset_index(drop=True)
    df_others = df[df["firstname"] != df["lastname"]].reset_index(drop=True)

    df_single["firstname"] = df_single["firstname"].str.lower()
    df_single["lastname"] = df_single["lastname"].str.lower()

    df_others["lastname"] = df_others["lastname"].str.lower()
    df_others["firstname"] = df_others["firstname"].str.lower()

    df_single = df_single.dropna()
    df_others = df_others.dropna()

    print("Shape of df_single:", df_single.shape)
    print("Shape of df_others:", df_others.shape)
    df_single.to_csv("./race_gender_data/df_single.csv")
    df_others.to_csv("./race_gender_data/df_others.csv")
