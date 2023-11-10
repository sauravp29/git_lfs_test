import pandas as pd
import numpy as np
import re
import pickle
from scipy.sparse import coo_matrix, hstack, csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .text_classification_utils import *


def create_amount_vars(df, val):
    data = df.copy()
    bins = pd.IntervalIndex.from_tuples(val)
    new_df = pd.cut(df['amount__c'], bins=bins)
    return pd.get_dummies(new_df)


def create_cur_balance_vars(df, val):
    data = df.copy()
    bins = pd.IntervalIndex.from_tuples(val)
    new_df = pd.cut(df['current_balance__c'], bins=bins)
    return pd.get_dummies(new_df)


def create_date_features(df):
    data = df.copy()
    bins = pd.IntervalIndex.from_tuples(
        [(0.97, 7.0), (7.0, 13.0), (13.0, 19.0), (19.0, 25.0), (25.0, 31.0)])
    new_df = pd.cut(df['date'].dt.day, bins=bins)
    return pd.get_dummies(new_df)


def day_type_variable(df):
    data = df.copy()
    new_df = (data['date'].dt.weekday > 5).replace({True: 1, False: 0})
    return new_df


def txn_type(df):
    data = df.copy()
    new_df = (data['amount__c'] > 0).replace({True: 1, False: 0})
    return new_df


def is_national_holiday(df, holiday_df):
    data = df.copy(deep=True)
    new_df = (df['date'].isin(holiday_df.date)).replace({True: 1, False: 0})
    new_df = new_df.astype('int64')
    return new_df


def txn_type(df):
    data = df.copy(deep=True)
    new_df = (df['tx_type__c'] == 'DEBIT').replace({True: 1, False: 0})
    return new_df


def sliding_description(filter_indi_desc, word_common):
    a_list = filter_indi_desc.split(' ')
    fin_lis = []
    for word in a_list:
        if word in word_common:
            fin_lis.append(word_common[word])
            continue
        flag_num = 0
        if word == '_number_':
            fin_lis.append('_number_')
            continue
        elif '_number_' in word:
            list_word = word.split('_')
            if len(list_word) == 5:
                word = list_word[2]
                flag_num = 2
            else:
                if list_word[0] == '':
                    word = list_word[2]
                    flag_num = 1
                else:
                    word = list_word[0]
                    flag_num = 3
        if flag_num <= 2 and flag_num > 0:
            fin_lis.append('_number_')
        if len(word) > 2:
            # print(flag_num)
            for q in range(len(word) - 2):
                window = q + 3
                while window < len(word) + 1:
                    fin_lis.append(word[q:window])
                    window += 1
            # print(fin_lis)
        else:
            fin_lis.append(word)
        if flag_num >= 2:
            fin_lis.append('_number_')
    return ' '.join(fin_lis)


def fix_string(x, word_common):
    fil_desc = re.sub(r'[^A-Za-z0-9 ]+', ' ', str(x)).lower().strip()
    fil_desc = re.sub(r'[0-9]+', '_number_', str(fil_desc))
    fil_desc = re.sub(' +', ' ', str(fil_desc))
    # print(fil_desc)
    final_desc = sliding_description(fil_desc, word_common)
    # fil_desc = fil_desc.replace( /  +/g, ' ' )
    return final_desc


def create_feature_matrix(
        conv_data,
        type_data,
        holiday_df,
        tfidf_pipeline=None):
    # conc_arr = conv_data[['tx_type__c', 'amount__c', 'current_balance__c']]
    # print(conv_data.dtypes)
    amount_vars = create_amount_vars(conv_data, val)
    balance_vars = create_cur_balance_vars(conv_data, balance_bin)
    date_vars = create_date_features(conv_data)

    day_type_vars = day_type_variable(conv_data)
    national_hol = is_national_holiday(conv_data, holiday_df)
    # bank_vars = bank_mapping(conv_data)
    # print(national_hol.value_counts())
    txn_type_data = txn_type(conv_data)
    # print(conv_data.dtypes)

    additional_df = pd.concat([amount_vars,
                               balance_vars,
                               date_vars,
                               date_vars,
                               national_hol,
                               txn_type_data],
                              axis=1)
    # print(additional_df.dtypes)
    nach_matrix = pd.concat([amount_vars, balance_vars, txn_type_data], axis=1)
    # additional_df = pd.concat([amount_vars, balance_vars], axis = 1)
    if type_data == 'training':
        tfidf_pipeline = Pipeline(
            [
                ('vect', CountVectorizer(
                    ngram_range=(
                        1, 2), max_features=5000)), ('tfidf', TfidfTransformer(
                            norm='l2', sublinear_tf=True))]).fit(
            conv_data.filtered_description)
    tfidf_matrix = tfidf_pipeline.transform(conv_data.filtered_description)
    stacking_matrix = csr_matrix(additional_df)
    # print(tfidf_matrix.shape, stacking_matrix.shape)
    final_matrix = hstack([tfidf_matrix, stacking_matrix])
    return final_matrix, nach_matrix


def create_feature_matrix_nach(
        conv_data,
        type_data,
        nach_matrix,
        tfidf_pipeline=None):
    # conc_arr = conv_data[['tx_type__c', 'amount__c', 'current_balance__c']]
    # amount_vars = create_amount_vars(conv_data, val)
    # balance_vars = create_cur_balance_vars(conv_data, balance_bin)
    # date_vars = create_date_features(conv_data)
    # day_type_vars = day_type_variable(conv_data)
    # national_hol = is_national_holiday(conv_data, holiday_df)
    # bank_vars = bank_mapping(conv_data)
    # txn_type_data = txn_type_data = txn_type(conv_data)
    # additional_df = pd.concat([amount_vars, balance_vars, date_vars, date_vars, national_hol, txn_type_data], axis = 1)
    # nach_matrix = pd.concat([amount_vars, balance_vars,txn_type_data])
    # additional_df = pd.concat([amount_vars, balance_vars], axis = 1)
    if type_data == 'training':
        tfidf_pipeline = Pipeline(
            [
                ('vect', CountVectorizer(
                    ngram_range=(
                        1, 2), max_features=5000)), ('tfidf', TfidfTransformer(
                            norm='l2', sublinear_tf=True))]).fit(
            conv_data.filtered_description)
    tfidf_matrix = tfidf_pipeline.transform(conv_data.filtered_description)
    stacking_matrix = csr_matrix(nach_matrix)
    # print(tfidf_matrix.shape, stacking_matrix.shape)
    final_matrix = hstack([tfidf_matrix, stacking_matrix])
    return final_matrix


def run_txn_type_model(df, model, tfidf_model, holiday_df, common_words):
    df = df.reset_index(drop=True)
    df['filtered_description'] = df['description'].apply(
        lambda x: fix_string(x, common_words))
    # print(df['date'])
    # df['date'] = pd.to_datetime(df['time_transformed'], format = '%Y-%m-%d').dt.date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    pred_matrix, nach_matrix = create_feature_matrix(
        df, 'testing', holiday_df, tfidf_pipeline=tfidf_model)
    y_pred = model.predict(pred_matrix.toarray())
    df['txn_classified_as'] = y_pred
    return df, nach_matrix


def run_nach_model(df, model, tfidf_model, nach_matrix):
    # df['filtered_description'] = df['description'].apply(lambda x: fix_string(x))
    # pred_matrix = create_feature_matrix(X_test, 'testing',tfidf_pipeline = tfidf_model)
    pred_matrix = create_feature_matrix_nach(
        df, 'testing', nach_matrix, tfidf_pipeline=tfidf_model)
    y_pred = model.predict(pred_matrix.toarray())
    df['nach_type'] = y_pred
    return df
