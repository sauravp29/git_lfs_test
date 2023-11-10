from .create_primitives import create_features

from multiprocessing import Pool
from functools import partial
import pandas as pd
import numpy as np
import os
import sys
import pickle
import cProfile
import pstats
import io
from pstats import SortKey


def get_schema_dict(SCHEMA_DICT_PATH):
    schema_dict = pickle.load(open(SCHEMA_DICT_PATH, 'rb'))
    return schema_dict


def hstack_df(list_of_df):
    df = pd.concat(list_of_df, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def find_count_sum(key):
    if ('COUNT' in key) or ('SUM') in key:
        return True
    else:
        return False


def get_stage_1_schema(schema_dict):
    sd1 = [i for i in schema_dict.keys() if ('intermediate' not in i[0])
           and ('window' not in i[0])]
    return sd1


def get_stage_2_schema(schema_dict):
    sd2 = [i for i in schema_dict.keys() if 'intermediate' in i[0]]
    return sd2


def get_stage_3_schema(schema_dict):
    sd3 = [i for i in schema_dict.keys() if ('intermediate' not in i[0])
           and ('window' in i[0])]
    return sd3


def create_stage_1_variables(dfi, dfa, schema_dict, sd1, grouped_dict):

    dfs_1 = {}
    for key_days in sd1:
        # print(key)
        filter_days = key_days[2]  # 30, 90 or 300 days,
        date_filter = dfa.created_date.values[0] - \
            np.timedelta64(filter_days, 'D')
        # Subseting by Applying time windows
        _dfi = dfi[dfi.date > date_filter]
        grouped_all_col_dict = {}
        for keys_ in grouped_dict:
            grouped_all_col_dict[keys_] = _dfi.groupby(keys_)
        inp_schema = schema_dict[key_days]
        records = {}
        for col, val in inp_schema:
            s_inp = inp_schema[col, val]
            try:
                if col:
                    dfin = grouped_all_col_dict[col].get_group(val)
                    # print(dfin.shape)
                    records.update(
                        create_features(
                            dfin, s_inp, (col, val), dfa))
                    # print(create_features(dfin, s_inp, (col, val), dfa))
                else:
                    records.update(
                        create_features(
                            _dfi, s_inp, (col, val), dfa))
            except BaseException:
                nan_arr = {}
                for col, prim_list in s_inp.items():
                    for prim, key in prim_list:
                        if prim in ['SUM', 'COUNT']:
                            nan_arr.update({key: 0})
                        else:
                            nan_arr.update({key: np.nan})
                records.update(nan_arr)
        # print(key_days)
        dfs_1[key_days] = pd.DataFrame.from_records(records, index=[1])
    return dfs_1


# def create_stage_2_variables_w1(u, _dfi, inp_schema, dfa, grouped_dict):

#     try:
#         #print(_dfi.shape)
#         grouped_all_col_dict = {}
#         for keys_ in grouped_dict:
#             grouped_all_col_dict[keys_] = _dfi.groupby(keys_)
#         records = {}
#         records['date'] = u
# #         pr = cProfile.Profile()
# #         pr.enable()
#         if inp_schema:
#             for col, val in inp_schema:
#                 #dfin = _dfi[_dfi.loc[:, col] == val] if col else _dfi
#                 s_inp = inp_schema[col, val]
#                 try:
#                     if col:
#                         dfin = grouped_all_col_dict[col].get_group(val)
#                         records.update(create_features(dfin, s_inp, (col, val), dfa))
#                     else:
#                         records.update(create_features(_dfi, s_inp, (col, val), dfa))
#                 except:
#                     nan_arr = {}
#                     for col, prim_list in s_inp.items():
#                         for prim, key in prim_list:
#                             if prim in ['SUM', 'COUNT']:
#                                 nan_arr.update({key : 0})
#                             else:
#                                 nan_arr.update({key : np.nan})
#                     records.update(nan_arr)

# #         pr.disable()
# #         s = io.StringIO()
# #         sortby = SortKey.CUMULATIVE
# #         ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# #         ps.print_stats()
# #         print(s.getvalue())
#     except Exception as e:
#         raise e
#     return records

def create_stage_2_variables(
        window_limits,
        dfi,
        inp_schema,
        dfa,
        grouped_dict,
        type_window):

    try:
        if type_window == 'window_30':
            l, u = window_limits
            condition_lower = dfi.date > l
            condition_upper = dfi.date <= u
            _dfi = dfi.loc[np.logical_and(condition_lower, condition_upper), :]
            records = {}
            records['date'] = u
        else:
            _dfi = dfi.copy()
            records = {}
            records['date'] = window_limits
        # print(_dfi.shape)
        grouped_all_col_dict = {}
        for keys_ in grouped_dict:
            grouped_all_col_dict[keys_] = _dfi.groupby(keys_)

#         pr = cProfile.Profile()
#         pr.enable()
        if inp_schema:
            for col, val in inp_schema:
                # dfin = _dfi[_dfi.loc[:, col] == val] if col else _dfi
                s_inp = inp_schema[col, val]
                try:
                    if col:
                        # print('in if')
                        dfin = grouped_all_col_dict[col].get_group(val)
                        records.update(
                            create_features(
                                dfin, s_inp, (col, val), dfa))
                    else:
                        # print('else here')
                        records.update(
                            create_features(
                                _dfi, s_inp, (col, val), dfa))
                except BaseException:
                    nan_arr = {}
                    for col, prim_list in s_inp.items():
                        for prim, key in prim_list:
                            if prim in ['SUM', 'COUNT']:
                                nan_arr.update({key: 0})
                            else:
                                nan_arr.update({key: np.nan})
                    records.update(nan_arr)

#         pr.disable()
#         s = io.StringIO()
#         sortby = SortKey.CUMULATIVE
#         ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#         ps.print_stats()
#         print(s.getvalue())
    except Exception as e:
        raise e
    return records

# def create_stage_2_variables(window_limits, dfi, inp_schema, dfa,
# grouped_dict):

#     try:
#         l, u = window_limits
#         condition_lower = dfi.date > l
#         condition_upper = dfi.date <= u
#         _dfi = dfi.loc[np.logical_and(condition_lower, condition_upper), :]
#         records = {}
#         records['date'] = u
#         if inp_schema:
#             for col, val in inp_schema:
#                 dfin = _dfi[_dfi.loc[:, col] == val] if col else _dfi
#                 s_inp = inp_schema[col, val]
#                 records.update(create_features(dfin, s_inp, (col, val), dfa))
#     except Exception as e:
#         raise e
#     return records


def create_stage_2_variables_wrapper(
        dfi,
        dfa,
        schema_dict,
        sd2,
        grouped_dict,
        internal_mp=False,
        n_jobs=2):
    #     pr = cProfile.Profile()
    #     pr.enable()
    first_txn_time = dfi.date.min()
    created_time = dfa.created_date.values[0]

    # import time
    window_arr_30 = np.arange(first_txn_time - np.timedelta64(30, 'D'),
                              created_time + 1, np.timedelta64(30, 'D'))
    window_arr_1 = pd.date_range(first_txn_time, created_time, freq='d')

    schema_w_3, schema_w_1 = None, None
    for each in sd2:
        if '_intermediate_30' in each[0]:
            schema_w_3 = each
        elif '_intermediate_1' in each[0]:
            schema_w_1 = each

    # print(schema_dict.get(schema_w_3))
#     stage_2_w_1_partial = partial(
#         create_stage_2_variables_fast,
#         dfi=dfi,
#         inp_schema=schema_dict.get(schema_w_1),
#         dfa=dfa,
#         grouped_dict = grouped_dict)

#     stage_2_w_3_partial = partial(
#         create_stage_2_variables,
#         dfi=dfi,
#         inp_schema=schema_dict.get(schema_w_3),
#         dfa=dfa,
#         grouped_dict = grouped_dict)

    # window_1_zip = zip(window_arr_1[:-1], window_arr_1[1:])
    window_3_zip = zip(window_arr_30[:-1], window_arr_30[1:])

    if internal_mp:

        with Pool(n_jobs) as p:
            # st = time.time()
            results_1 = p.map(stage_2_w_1_partial, window_1_zip)
        with Pool(n_jobs) as p:
            # st = time.time()

            results_3 = p.map(stage_2_w_3_partial, window_3_zip)
    else:
        results_1 = []
        results_3 = []
        date_grouped_var = dfi.groupby('date')
        empt_df = pd.DataFrame(columns=dfi.columns)
        for each in window_arr_1:
            try:
                _dfi = date_grouped_var.get_group(each)
            except BaseException:
                _dfi = empt_df

            results_1.append(
                create_stage_2_variables(
                    window_limits=each,
                    dfi=_dfi,
                    inp_schema=schema_dict.get(schema_w_1),
                    dfa=dfa,
                    grouped_dict=grouped_dict,
                    type_window='window_1'))
        # pd.DataFrame(results_1).to_csv('window_1.csv', index = False)
        for each in window_3_zip:
            results_3.append(
                create_stage_2_variables(
                    window_limits=each,
                    dfi=dfi,
                    inp_schema=schema_dict.get(schema_w_3),
                    dfa=dfa,
                    grouped_dict=grouped_dict,
                    type_window='window_30'))

#     pr.disable()
#     s = io.StringIO()
#     sortby = SortKey.CUMULATIVE
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     print(s.getvalue())

    df_window_1 = pd.DataFrame.from_records(results_1)
    df_window_3 = pd.DataFrame.from_records(results_3)

    return df_window_1, df_window_3


def create_stage_3_variables(df_window_1, df_window_3, dfa, schema_dict, sd3):

    dfs_3 = {}
    # print(sd3)
    # print(df_window_1.columns)
    # print(df_window_3.columns)
    for k in sd3:
        filter_days = k[2]
        date_filter = dfa.created_date.values[0] - \
            np.timedelta64(filter_days, 'D')

        if '_window_1' in k[0]:
            _dfi = df_window_1
        elif '_window_30' in k[0]:
            _dfi = df_window_3

        _dfi = _dfi[_dfi.date > date_filter]  # Cutoff date

        inp_schema = schema_dict[k]
        rec = {}

        for col, val in inp_schema:
            dfin = _dfi[_dfi.loc[:, col] == val] if col else _dfi
            s_inp = inp_schema[col, val]
            feats_out = create_features(dfin, s_inp, (col, val), dfa)
            rec.update(feats_out)

        dfs_3[k] = pd.DataFrame.from_records(rec, index=[1])
    return dfs_3


def create_dfs_features(
        dfi,
        dfa,
        schema_dict,
        mapper_dict_keys,
        internal_mp=False,
        n_jobs=2):

    #     from .dfs_additional_variables_creation import create_additional_variables
    #     dfi, dfa = create_additional_variables(data_df)

    if isinstance(schema_dict, str):
        schema_dict = get_schema_dict(schema_dict)
    elif not isinstance(schema_dict, dict):
        raise Exception('Schema Dict should be of Dict type or Its path')

    # Compute Stage 1, Most recent 30, 90 and 300 days aggregation on the the
    # data.
    # print(grouped_all_col_dict)
    dfs_stage_1_features = create_stage_1_variables(
        dfi, dfa, schema_dict, get_stage_1_schema(schema_dict), mapper_dict_keys)
    # print(dfs_stage_1_features.keys())
    df_simple_30 = dfs_stage_1_features.get(('_features_30', None, 30, 1))
    df_simple_300 = dfs_stage_1_features.get(('_features_300', None, 300, 1))
    df_simple_90 = dfs_stage_1_features.get(('_features_90', None, 90, 1))

    # Compute Stage 2, Aggregate entity by Applying a window of 1 day and 1 month. (stage 1 is only for most recent)
    # Here single identifer will still have multiple rows, does it is not
    # outputed directly and is aggregated throught stage 3.
    dfs_stage_2_features = create_stage_2_variables_wrapper(
        dfi,
        dfa,
        schema_dict,
        get_stage_2_schema(schema_dict),
        mapper_dict_keys,
        internal_mp,
        n_jobs)

    df_window_1 = dfs_stage_2_features[0]
    df_window_3 = dfs_stage_2_features[1]

    dfs_stage_3_features = create_stage_3_variables(
        df_window_1, df_window_3, dfa, schema_dict, get_stage_3_schema(schema_dict))

    df_window_1_features_30 = dfs_stage_3_features.get(
        ('_window_1_features_30', 1, 30, 2))
    df_window_1_features_90 = dfs_stage_3_features.get(
        ('_window_1_features_90', 1, 90, 2))
    df_window_1_features_300 = dfs_stage_3_features.get(
        ('_window_1_features_300', 1, 300, 2))
    df_window_30_features_300 = dfs_stage_3_features.get(
        ('_window_30_features_300', 30, 300, 2))

    final_df = hstack_df(
        [df_simple_30,
         df_simple_300,
         df_simple_90,
         df_window_1_features_30,
         df_window_1_features_90,
         df_window_1_features_300,
         df_window_30_features_300])

    return final_df
