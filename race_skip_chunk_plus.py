import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import os
from ethnicolr2 import pred_fl_last_name, pred_fl_full_name
import warnings


# Define a function to process a chunk and handle errors
def process_chunk(chunk, prediction_function, lname_col=None, fname_col=None):
    while True:
        try:
            # Attempt to process the entire chunk
            if prediction_function == pred_fl_full_name:
                return prediction_function(
                    chunk, lname_col=lname_col, fname_col=fname_col
                )
            elif prediction_function == pred_fl_last_name:
                return prediction_function(chunk, lname_col=lname_col)

        except IndexError as e:
            print(f"IndexError in chunk: {e}. Identifying problematic row.")

            # Find the problematic row by running each row individually
            for idx, row in chunk.iterrows():
                try:
                    if prediction_function == pred_fl_full_name:
                        prediction_function(
                            pd.DataFrame([row]),
                            lname_col=lname_col,
                            fname_col=fname_col,
                        )
                    elif prediction_function == pred_fl_last_name:
                        prediction_function(pd.DataFrame([row]), lname_col=lname_col)
                except IndexError:
                    # Log and remove the problematic row, then stop individual row processing
                    print(f"Deleting row with index: {idx}")
                    chunk = chunk.drop(idx)
                    break  # Exit the loop after finding and removing one problematic row

        except Exception as e:
            print(f"Error processing chunk: {e}")
            # Handle other exceptions by setting predictions to None
            return pd.DataFrame({col: [None] * len(chunk) for col in chunk.columns})


# Modify process_in_chunks to save each chunk independently
def process_in_chunks(
    df,
    chunk_size,
    prediction_function,
    lname_col=None,
    fname_col=None,
    filename_prefix="",
):
    with Pool(processes=cpu_count()) as pool:  # Use all available CPU cores
        for chunk_index, start in enumerate(range(0, len(df), chunk_size), start=1):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]

            # Apply prediction function on the chunk asynchronously
            result = pool.apply_async(
                process_chunk, (chunk, prediction_function, lname_col, fname_col)
            )

            # Get the processed result
            pred_chunk = result.get()

            # Save each chunk to a separate CSV file with a unique name
            pred_chunk.to_csv(
                f"./race_gender_data/{filename_prefix}_chunk_{chunk_index}.csv",
                index=False,
            )
            print(f"Processed and saved chunk {chunk_index} from row {start} to {end}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    df_single = pd.read_csv("./race_gender_data/df_single.csv").iloc[:, 1:]
    df_others = pd.read_csv("./race_gender_data/df_others.csv").iloc[:, 1:]
    print(df_single.shape, df_others.shape)

    # # Process and save df_single in chunks of 10000 rows for last name prediction
    # process_in_chunks(
    #     df_single,
    #     chunk_size=10000,
    #     prediction_function=pred_fl_last_name,
    #     lname_col="lastname",
    #     filename_prefix="df_single",
    # )

    # Process and save df_others in chunks of 300000 rows
    process_in_chunks(
        df_others,
        chunk_size=500000,
        prediction_function=pred_fl_full_name,
        lname_col="lastname",
        fname_col="firstname",
        filename_prefix="df_others",
    )
