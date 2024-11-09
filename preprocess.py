import pandas as pd
import numpy as np
import re


def is_valid_name(name):
    return isinstance(name, str) and name.isalpha()


# Function to process the names based on the conditions
def process_names(row):
    lastname = row["lastname"]

    # Filter out rows where lastname is not a string
    if not isinstance(lastname, str):
        return row  # Skip processing for this row

    # Case 1: If lastname is a single name, set both firstname and lastname to lastname
    if " " not in lastname:
        row["firstname"] = lastname
        row["lastname"] = lastname

    # Case 2: If lastname has multiple spaces, use the last space to separate
    elif len(lastname.split()) > 2:
        row["firstname"] = re.sub(
            r" (\S+)$", "", lastname
        )  # Everything before the last space
        row["lastname"] = re.sub(
            r".* (\S+)$", r"\1", lastname
        )  # Only the last word after the last space

    # Case 3: If lastname has only one space, split by that space
    else:
        parts = lastname.split(" ", 1)
        row["firstname"] = parts[0]
        row["lastname"] = parts[1]

    return row


if __name__ == "__main__":
    # raw data
    df = pd.read_csv("username.csv")
    df = df.iloc[:, 1:]
    df = df.drop_duplicates()
    df["firstname"] = df["firstname"].str.strip()
    df["lastname"] = df["lastname"].str.strip()
    df_with_dash = df[df["firstname"] == "-"]
    df_without_dash = df[df["firstname"] != "-"]
    # Apply the function to each row
    df_with_dash = df_with_dash.apply(process_names, axis=1).reset_index(drop=True)
    df = pd.concat([df_with_dash, df_without_dash], axis=0)
    df = df[df["firstname"].apply(is_valid_name) & df["lastname"].apply(is_valid_name)]

    df["lastname"] = df["lastname"].str.lower()
    df["firstname"] = df["firstname"].str.lower()
    # trim leading and aftering extra space fo fullname in df
    df["fullname"] = np.where(
        df["firstname"] == df["lastname"],
        df["firstname"],
        df["firstname"] + " " + df["lastname"],
    )
    df["fullname"] = df["fullname"].str.strip()
    # remove invaild fullname
    df = df[df["fullname"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    # Remove rows with NA values
    df = df.dropna()

    print("Shape of df:", df.shape)
    df.to_csv("./race_gender_data/df.csv")
