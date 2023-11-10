from itertools import combinations, permutations
import pandas as pd
import numpy as np


def filter_txn_by_FILTER_DAYS_MAX(txn_df, FILTER_DAYS_MAX):
    inner_df = txn_df.copy(deep=True)
    filter_date = inner_df['application_created_date'].iloc[0] - \
        pd.Timedelta(days=FILTER_DAYS_MAX)
    inner_df = inner_df[inner_df['date'] >= filter_date]
    inner_df = inner_df.reset_index(drop=True)

    return inner_df


def divide(a, b):
    if b == 0:
        return np.nan
    else:
        return a / b


def count_feat(data, list_of_cat_names, classification_type):
    if isinstance(data, list):
        # print(data[0])
        return data[0]
    dict_ = {}
    # print(data[classification_type].value_counts())
    for i, j in zip(data[classification_type].value_counts(
    ).index, data[classification_type].value_counts()):
        # print(i, j)
        dict_[f'count_of_{i}'] = j
    for cat in list_of_cat_names:
        if f'count_of_{cat}' not in dict_:
            dict_[f'count_of_{cat}'] = 0
            if 'all' in cat:
                dict_[f'count_of_{cat}'] = data.shape[0]
    return dict_


def create_frac_feats(
        txn_df,
        list_of_cat_names,
        feat_ratio_dict,
        classification_type):
    txn_df_filtered = txn_df.copy(True)
    # print(feature_list)
    # print(feat_ratio_dict)
    final_feature_ratio_dict = {}
    feature_matrix = count_feat(
        txn_df_filtered,
        list_of_cat_names,
        classification_type)
    # print(feature_matrix)
    for key, cols in feat_ratio_dict.items():
        if (cols[0] in feature_matrix) and (cols[1] in feature_matrix):
            final_feature_ratio_dict[key] = divide(
                feature_matrix[cols[0]], feature_matrix[cols[1]])
    return final_feature_ratio_dict
