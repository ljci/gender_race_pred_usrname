import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import os
from ethnicolr2 import pred_fl_last_name, pred_fl_full_name
import warnings


# Define a function to process a chunk and handle errors
def process_chunk(chunk, prediction_function, lname_col=None, fname_col=None):
    last_deleted_index = None  # Track the last deleted row's index

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

            # Process only rows after the last deleted row to find the next problematic row
            for idx, row in chunk.iterrows():
                if last_deleted_index is not None and idx <= last_deleted_index:
                    continue  # Skip rows processed in previous error-check

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
                    # Log and remove the problematic row, update last deleted index
                    print(f"Deleting row with index: {idx}")
                    chunk = chunk.drop(idx)
                    last_deleted_index = idx
                    break  # Exit after removing the problematic row and rerun the chunk

        except Exception as e:
            print(f"Error processing chunk: {e}")
            # Handle other exceptions by setting predictions to None
            return pd.DataFrame({col: [None] * len(chunk) for col in chunk.columns})


# Modify process_in_chunks to start from a specific chunk and save each chunk independently
def process_in_chunks(
    df,
    chunk_size,
    prediction_function,
    lname_col=None,
    fname_col=None,
    filename_prefix="",
    start_chunk=1,  # Default to process from the first chunk; can be set to skip chunks
):
    with Pool(processes=cpu_count()) as pool:  # Use all available CPU cores
        total_chunks = (
            len(df) + chunk_size - 1
        ) // chunk_size  # Calculate total number of chunks

        for chunk_index in range(
            start_chunk, total_chunks + 1
        ):  # Start from the specified chunk
            start = (chunk_index - 1) * chunk_size
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
    df_others = pd.read_csv("./race_gender_data/df_others.csv").iloc[:, 1:]
    print("df_others shape:", df_others.shape)

    # Process and save df_others in chunks, starting from the 3rd chunk
    process_in_chunks(
        df_others,
        chunk_size=500000,
        prediction_function=pred_fl_full_name,
        lname_col="lastname",
        fname_col="firstname",
        filename_prefix="df_others",
        start_chunk=3,  # Start processing from chunk 3
    )
