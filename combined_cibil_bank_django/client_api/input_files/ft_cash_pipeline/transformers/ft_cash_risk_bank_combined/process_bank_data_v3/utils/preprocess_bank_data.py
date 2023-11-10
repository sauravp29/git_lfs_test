import pandas as pd
from ..exceptions import *


def transform_timestamp(
        inner_df,
        created_date_format,
        txn_date_format,
        created_date_format_unit,
        txn_date_format_unit, batch_prepoc=False):
    """
    Converts the timestamp value stored in "txn_date" column into "time_transformed
    param: inner_df: The dataframe containing txn_data
    return: transformed_df : Contains the a new column "time_transformed", which is timestamp converted into date.
    """
    df = inner_df.copy(deep=True)
    df = df[df['date'].notna()]
    # print(df.columns)
    if batch_prepoc:
        if txn_date_format == 'epoch':
            df['date'] = pd.to_numeric(df['date'], errors='coerce')
            df = df[(df['date'] >= 0) & (df['date'].notna())]
            df['time_transformed'] = pd.to_datetime(
                df["date"], unit=txn_date_format_unit, errors='coerce').dt.date

        else:
            df['time_transformed'] = pd.to_datetime(
                df["date"], format=txn_date_format, errors='coerce').dt.date
        df['date'] = df['time_transformed']
    else:
        if txn_date_format == 'epoch':
            df['date'] = pd.to_numeric(df['date'], errors='coerce')
            df = df[(df['date'] >= 0) & (df['date'].notna())]
            df['time_transformed'] = pd.to_datetime(
                df["date"], unit=txn_date_format_unit).dt.date
        else:
            df['time_transformed'] = pd.to_datetime(
                df["date"], format=txn_date_format).dt.date

    df["time_transformed"] = pd.to_datetime(
        df["time_transformed"], format="%Y-%m-%d")

    if created_date_format == 'epoch':
        df['application_created_date'] = pd.to_numeric(
            df['application_created_date'], errors='raise')
        df['application_created_date'] = pd.to_datetime(
            df["application_created_date"], unit=created_date_format).dt.date

    else:
        if batch_prepoc:
            df['application_created_date'] = pd.to_datetime(
                df["application_created_date"],
                format=created_date_format,
                errors='coerce').dt.date
        else:
            df['application_created_date'] = pd.to_datetime(
                df["application_created_date"], format=created_date_format).dt.date

    df["application_created_date"] = pd.to_datetime(
        df["application_created_date"], format="%Y-%m-%d")
    df['date'] = df['time_transformed']
    df['temp_index'] = df.index
#     df = df.sort_values(
#         by=['time_transformed', 'temp_index'], ascending=[True, True])
    df = df.drop(columns='temp_index')
    df = df.reset_index(drop=True)
    return df


def filter_within_cutoff(inner_df):
    """
    Filters the messages based on their time value.
    param: inner_df: The dataframe containing txn_data
    return: filtered_df : Contains the message where time is between the created_date
    """
    df = inner_df.copy(deep=True)
    df = df[df["time_transformed"] <= df['application_created_date']]
    df = df.reset_index(drop=True)
    return df


def get_preprocessed_bank_df(
        txn_df,
        created_date_format,
        txn_date_format,
        created_date_format_unit,
        txn_date_format_unit, batch_prepoc=False):
    """
    1. Converts the jsonfile into DataFrame by invoking convert_to_DataFrame()
    2. Renames the columns by invoking change_column_name()
    3. Filters the messages between the cutoff filters by invoking filter_within_cutoff()
    4. Removes duplicates messages by invoking remove_duplicated_messages()
    5. Reset index
    params: json_dict: The input txn json
    returns: dups_removed_df: Processed DataFrame
    """
    try:
        df = txn_df.copy()
        df = df.dropna()

        time_transformed_df = transform_timestamp(
            df,
            created_date_format,
            txn_date_format,
            created_date_format_unit,
            txn_date_format_unit, batch_prepoc)
        time_transformed_df = time_transformed_df[~pd.isnull(
            time_transformed_df['time_transformed'])]

        filter_df = filter_within_cutoff(time_transformed_df)

        if filter_df.shape[0] == 0:

            raise RawBankingNoDataAfterFilter(
                'No txn left after filtering the data')

        return filter_df

    except Exception as e:
        raise RawBankingPreprocessFailedException(
            'error occurred while preprocessing raw txn json: ' + str(e))
