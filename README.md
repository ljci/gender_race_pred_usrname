
- [Logic](#logic)
  - [General line](#general-line)
  - [Code line](#code-line)
    - [Preprocess](#preprocess)
    - [Race prediction](#race-prediction)
- [Reference](#reference)

# Logic

## General line

read username.csv →df → clean df → predict gender: df_gender→ split df_single, df_others -> predict race -> race_gender_full_others.csv, race_gender_last_others.csv -> concat two csv-> final csv

## Code line

### Preprocess

`preprocess.py`→`gender.py`→`split.py`→ `race_full.py`, `race_last.py`<br>
> output: race_gender_full_others.csv, race_gender_last_others.csv

`preprocess.py`: read and clean raw df<br>
>output: clean df

`gender.py`: predict gender on clean df<br>
>output: df_gender

`split.py`: split clean df into single name(e.g yass, john) and others(e.g john whiteson)<br>
> output: df_single, df_others<br>
> caution: df_single, firstname =lastname. e.g firstname: yass, lastname: yass, as it is a single name

### Race prediction

`race_skip_chunk.py`: set chunk size to run model on chunks of whole dataset. Once the chunk has index error, skip this chunk and run next.

`race_skip_chunk_plus.py`:
Set the chunk size to run pre-trained model(optimize the cpu usage).When chunk has index-error, find the rows that lead to index error, delete that rows and print out which rows is being delete from dataframe, after deleting rows, rerun chunk. repeat until each chunk can be successfully run

`race_skip_chunk_plus_rest.py`: upgraded method on `race_skip_chunk_plus.py` but only used for the rest chunks of df_others(in this case, from chunk 3). When rerunning the chunk again after deleting the previous row that cause indexerror, and it has indexerror,  each row in the chunk is processed individually to find problematic rows, for this time, we only need to run row-by-row process on those rows after the previous problmetics row that is deleted in previous, so we don't need to run row-by-row process on everyrow, only on those rows after previous problematic rows.
row-by row process is simply run model on each row instead of the chunk to find the row lead to index error of model.

# Reference

1. [Name-Gender-Predictor](https://github.com/imshibl/Name-Gender-Predictor?tab=readme-ov-file#model-training)
2. [Name-Race-Predictor](https://github.com/appeler/ethnicolr2?tab=readme-ov-file)
3. [Name-Race-Predictor](https://github.com/appeler/ethnicolr_v2?tab=readme-ov-file)
