from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
from . import CONSTANT
import re
import time
from .exceptions import *
import warnings

from .utils.preprocess_bank_data import get_preprocessed_bank_df
from .utils.run_text_cat_models import run_txn_type_model, run_nach_model
from .utils.create_additional_dfs_columns import create_additional_columns
from .utils.fraction_features_create import create_frac_feats
from .utils.get_required_columns_pre_dfs import get_required_columns


def sort_txns_by_date(df):
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'idx'  # index
    df.sort_values(by=['date', 'idx'], ascending=[  # index
                   True, True], inplace=True)
    return df


def deduce_from_amount(df, deduce_type_from_amount):
    if deduce_type_from_amount:
        try:
            df['type'] = df['amount'].apply(
                lambda x: 'credit' if float(x) >= 0 else "debit")
            df['amount'] = df['amount'].apply(
                lambda x: abs(float(x)))
        except Exception as e:
            raise e
    else:
        try:
            df['type'] = df['type'].str.upper()
        except BaseException:
            raise Exception('Type of txn should be present')
    return df


def formatter(df, mapper_functions_dict):
    for each_col in df.columns:
        try:
            if each_col in mapper_functions_dict.keys():
                df[each_col] = df[each_col].apply(
                    lambda x: mapper_functions_dict[each_col](x))
        except Exception as e:
            raise Exception(
                "Got Error while applying FORMATTER_FUNCTION on " +
                each_col +
                "\n Error : " +
                str(e))
    return df


def filter_txn_by_FILTER_DAYS_MAX(inner_df, FILTER_DAYS_MAX):

    if FILTER_DAYS_MAX > 300:
        FILTER_DAYS_MAX = 300
        warnings.warn("FILTER_MAX_DAYS can not be greater than 300")
    filter_date = inner_df['application_created_date'] - \
        pd.Timedelta(days=FILTER_DAYS_MAX)
    inner_df = inner_df[inner_df['time_transformed'] >= filter_date]
    inner_df = inner_df.reset_index(drop=True)

    return inner_df


def split_frac_features(features, list_of_txn):
    temp_feat = []
    feat_ratio_dict = {}
    day_dict = CONSTANT.FRACTION_DAY_DICT
    for feat in features:
        if not feat.startswith('frac'):
            temp_feat.append(feat)
        else:
            feat_ratio_dict[feat] = []
            frac, feat_1, feat_2 = feat.split('__')
            temp_feat.append(feat_1)
            feat_2, days_since = feat_2.split('_features_')
            temp_feat.append(feat_2)
            if feat_1[9:] in list_of_txn and feat_2[9:] in list_of_txn:
                day_dict['txn_classified_as'][int(days_since)].update({
                    feat: [feat_1, feat_2]})
            else:
                day_dict['nach_type'][int(days_since)].update(
                    {feat: [feat_1, feat_2]})

    return temp_feat, day_dict


def preprocess_banking_data(
        txn_df,
        MAPPER_FORMATTER_FUNCTION_DICT,
        CONFIG_DICT,
        df_subset_custom_function,
        FILTER_MAX_DAYS, batch_prepoc=False):

    try:
        deduce_type_from_amount = CONFIG_DICT['deduce_type_from_amount']
        created_date_format = CONFIG_DICT['created_date_format']

        if created_date_format == 'epoch':
            created_date_format_unit = CONFIG_DICT['created_date_format_unit']
        else:
            created_date_format_unit = None

        txn_date_format = CONFIG_DICT['transaction_date_format']
        if txn_date_format == 'epoch':
            txn_date_format_unit = CONFIG_DICT['transaction_date_format_unit']
        else:
            txn_date_format_unit = None

    except Exception as e:
        raise RawBankingPreprocessFailedException(
            CONSTANT.CONFIG_DICT_ERROR +
            str(e))
    try:
        # print(txn_df.columns)
        preprocessed_banking_df = get_preprocessed_bank_df(
            txn_df,
            created_date_format,
            txn_date_format,
            created_date_format_unit,
            txn_date_format_unit,
            batch_prepoc)

        if FILTER_MAX_DAYS:
            preprocessed_banking_df = filter_txn_by_FILTER_DAYS_MAX(
                preprocessed_banking_df, FILTER_MAX_DAYS)

        if MAPPER_FORMATTER_FUNCTION_DICT:
            preprocessed_banking_df = formatter(
                preprocessed_banking_df, MAPPER_FORMATTER_FUNCTION_DICT)

        # TODO: change name to df wise custom function and column wise custom
        # function
        if df_subset_custom_function:
            preprocessed_banking_df = df_subset_custom_function(
                preprocessed_banking_df)
        if preprocessed_banking_df.empty:
            raise RawBankingPreprocessFailedException(
                CONSTANT.NO_DATA_REMAINING_ERROR)

        return preprocessed_banking_df

    except Exception as e:
        raise RawBankingPreprocessFailedException(
            CONSTANT.PREPROCESS_ERROR +
            str(e))


def text_classification(
        txn_df,
        text_classification_model,
        tfidf_pipeline_txn,
        nach_model,
        tfidf_model_nach,
        holiday_df,
        words_common):

    try:
        classified_txn_data_df, nach_matrix = text_cat_out, nach_matrix = run_txn_type_model(
            txn_df, text_classification_model, tfidf_pipeline_txn, holiday_df, words_common)

    except Exception as e:
        raise TextClassificationFailedException(
            CONSTANT.TEXTCAT_ERROR + str(e))

    try:
        nach_classified_txn_data_df = run_nach_model(
            classified_txn_data_df, nach_model, tfidf_model_nach, nach_matrix)
        nach_classified_txn_data_df = nach_classified_txn_data_df.rename(
            columns={
                'amount__c': 'amount',
                'current_balance__c': 'balance',
                'tx_type__c': 'type'})
        return nach_classified_txn_data_df

    except Exception as e:
        raise NachClassificationFailedException(
            CONSTANT.NACH_CLASSIFY_ERROR + str(e))


def create_pre_dfs_data(
        txn_classified_df,
        holiday_df,
        features_list,
        column_dict=None,
        BATCH_PREPROCESS=False):
    try:
        if not BATCH_PREPROCESS:
            mapper_dict_keys = get_required_columns(features_list)
            final_pre_dfs_df = create_additional_columns(txn_classified_df, txn_classified_df[[
                                                         'application_id', 'application_created_date']], holiday_df, mapper_dict_keys, column_dict)
            return final_pre_dfs_df, final_pre_dfs_df[[
                'application_id', 'application_created_date']], mapper_dict_keys
        else:
            final_pre_dfs_df = create_additional_columns(txn_classified_df, txn_classified_df[[
                                                         'application_id', 'application_created_date']], holiday_df, column_dict)
            return final_pre_dfs_df, final_pre_dfs_df[[
                'application_id', 'application_created_date']], None
    except Exception as e:
        raise AdditionalColumnCreationFailedException(
            CONSTANT.DFS_DATA_CREATION_ERROR + str(e))


def create_fraction_features(
        banking_df,
        fraction_features,
        list_of_txn_cats,
        list_of_nach_cats):
    try:
        dict_ = {}
        list_of_txn_cats = list_of_txn_cats + ['all_nach']
        list_of_nach_cats = list_of_nach_cats + ['all_txn']
        temp_feat, day_type_wise_dict = split_frac_features(
            fraction_features, list_of_txn_cats)
        for type_of_class in day_type_wise_dict:
            for day_ in (day_type_wise_dict[type_of_class]):
                df_filtered = filter_txn_by_FILTER_DAYS_MAX(banking_df, day_)
                if type_of_class == 'nach_type':
                    dict_.update(
                        create_frac_feats(
                            df_filtered,
                            list_of_nach_cats,
                            day_type_wise_dict[type_of_class][day_],
                            type_of_class))
                    # print(dict_)
                else:
                    dict_.update(
                        create_frac_feats(
                            df_filtered,
                            list_of_txn_cats,
                            day_type_wise_dict[type_of_class][day_],
                            type_of_class))

        return dict_

    except Exception as e:
        raise PipelineFourDataPrepocException(
            CONSTANT.FRACTION_FEATURE_ERROR + str(e))
