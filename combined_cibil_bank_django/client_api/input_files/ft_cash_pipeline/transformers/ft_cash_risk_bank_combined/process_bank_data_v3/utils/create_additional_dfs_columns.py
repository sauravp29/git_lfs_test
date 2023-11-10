import pandas as pd
import datetime
import numpy as np

# def day_month_year(dfi):
#     df = dfi.copy()

#     df['date_day'] = df['date'].dt.day
#     df['date_month'] = df['date'].dt.month
#     df['date_weekday'] = df['date'].dt.weekday
#     return df


def day_type_col(dfi):
    df = dfi.copy()
    df['day_type'] = df['date'].dt.dayofweek.replace(
        to_replace=[
            0, 1, 2, 3, 4, 5, 6], value=[
            'wd', 'wd', 'wd', 'wd', 'we', 'we', 'we'])
    return df


def datediff_col(dfi):
    df = dfi.copy()
    df['datediff_txn'] = (
        df['date'] -
        df['application_created_date']).dt.days.astype(float)
    df['datediff_txn'] = np.abs(df['datediff_txn'])
    return df


def first_7_days_col(dfi):
    df = dfi.copy()
    df['first_7_days'] = (df['date'].dt.day <= 7).replace(
        {True: '1', False: '0'})
    return df


def last_7_days_col(dfi):
    df = dfi.copy()
    df['last_7_days'] = (df['date'].dt.day > 24).replace(
        {True: '1', False: '0'})
    return df


def range_cols(dfi, x):
    df = dfi.copy()
    split_lis = x.split('_')
    name = '_'.join(split_lis[:-1])
    amt_rng = split_lis[-1]
    rng1 = amt_rng.split('-')[0]
    rng2 = amt_rng.split('-')[1]

    if rng1 == 'l':
        df[x] = (df[name] <= 0).replace(
            {True: '1', False: '0'})
    elif rng2 == '':
        df[x] = (df[name] > int(rng1)).replace(
            {True: '1', False: '0'})
    else:
        df[x] = ((df[name] > int(rng1)) & (df[name] <= int(rng2))).replace(
            {True: '1', False: '0'})

    return df


def create_combined_cols(dfi, x):
    df = dfi.copy()
    x = str(x)
    x_div = x.split('-')
    df[x] = df['-'.join(x_div[:-1])].astype(
        str).str.cat(df[x_div[-1]].astype(str), sep='-')
    return df


def mark_national_holiday(dfi, holiday_df, x):
    df = dfi.copy(deep=True)
    df[x] = 0
    df.loc[df.date.isin(holiday_df.date), x] = 1
    return df


def is_redundant(dfi):
    df = dfi.copy(deep=True)
    ilocs = set()
    i = 0
    df_dict_list = df.reset_index(
    )[['application_id', 'type', 'amount', 'date']].to_dict('records')

    while i < (len(df_dict_list) - 1):
        row_i = df_dict_list[i]
        row_i_next = df_dict_list[i + 1]
        if ((row_i['type'] != row_i_next['type'])
            and (row_i['amount'] == row_i_next['amount'])
                and (row_i['date'] == row_i_next['date'])
                and (row_i['application_id'] == row_i_next['application_id'])):
            ilocs.add(i)
            ilocs.add(i + 1)
            i += 2
        else:
            i += 1
    ilocs = list(ilocs)
    return ilocs


def create_additional_columns(
        dfi,
        dfa,
        holiday_df,
        mapper_list_keys=None,
        column_dict=None):
    # print(mapper_list_keys)
    dfi['is_redundant'] = 0
    dfi.loc[is_redundant(dfi), 'is_redundant'] = 1
    # print(dfi.isna().sum())
    # dfi = dfi.dropna()
    def_mapper_dict = {
        'datediff_txn': datediff_col,
        'day_type': day_type_col,
        'is_national_holiday': mark_national_holiday,
        'first_7_days': first_7_days_col,
        'last_7_days': last_7_days_col,
        'amount_0-1000': range_cols,
        'amount_1000-5000': range_cols,
        'amount_5000-10000': range_cols,
        'amount_10000-20000': range_cols,
        'amount_20000-50000': range_cols,
        'amount_50000-': range_cols,
        'balance_l-0': range_cols,
        'balance_0-10000': range_cols,
        'balance_10000-20000': range_cols,
        'balance_20000-50000': range_cols,
        'balance_50000-100000': range_cols,
        'balance_100000-': range_cols,
        'day_type-type': create_combined_cols,
        'nach_type-type': create_combined_cols,
        'is_national_holiday-nach_type': create_combined_cols,
        'is_national_holiday-type': create_combined_cols,
        'is_national_holiday-txn_classified_as': create_combined_cols,
        'nach_type-txn_classified_as': create_combined_cols,
        'day_type-txn_classified_as': create_combined_cols,
        'is_redundant-nach_type': create_combined_cols,
        'is_redundant-txn_classified_as': create_combined_cols,
        'first_7_days-txn_classified_as': create_combined_cols,
        'last_7_days-txn_classified_as': create_combined_cols,
        'amount_0-1000-txn_classified_as': create_combined_cols,
        'amount_1000-5000-txn_classified_as': create_combined_cols,
        'amount_5000-10000-txn_classified_as': create_combined_cols,
        'amount_10000-20000-txn_classified_as': create_combined_cols,
        'amount_20000-50000-txn_classified_as': create_combined_cols,
        'amount_50000--txn_classified_as': create_combined_cols,
        'balance_l-0-txn_classified_as': create_combined_cols,
        'balance_0-10000-txn_classified_as': create_combined_cols,
        'balance_10000-20000-txn_classified_as': create_combined_cols,
        'balance_20000-50000-txn_classified_as': create_combined_cols,
        'balance_50000-100000-txn_classified_as': create_combined_cols,
        'balance_100000--txn_classified_as': create_combined_cols}

    if column_dict is not None:
        mapper_dict.update(column_dict)

    if mapper_list_keys:
        mapper_dict = {
            k: v for k,
            v in def_mapper_dict.items() if k in mapper_list_keys +
            ['datediff_txn']}
    else:
        mapper_dict = def_mapper_dict

    dfi['date'] = pd.to_datetime(dfi['date'], format='%Y-%m-%d')
    for col_name in mapper_dict:
        if '-' in col_name:
            dfi = mapper_dict[col_name](dfi, col_name)
        elif col_name == 'is_national_holiday':
            dfi = mapper_dict[col_name](dfi, holiday_df, col_name)
        else:
            dfi = mapper_dict[col_name](dfi)
        # print(dfi.columns)
    return dfi
