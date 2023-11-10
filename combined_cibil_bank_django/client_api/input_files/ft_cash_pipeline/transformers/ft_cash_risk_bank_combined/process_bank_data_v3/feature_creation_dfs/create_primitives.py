# to be run per filter per column
# entropy to be checked for nulls
# std with n-1 or std with n in the denominator.sample_skewness or population_skewness
# see if the linear regression is standardized
import scipy.stats
import statistics
import numpy as np
import re
import pandas as pd
import os
import sys
import pickle
import time
import warnings
from sklearn.linear_model import LinearRegression


def compute_primitive(df, time_index, prim, computed_resources, created_time):
    """
    Compute primitive by using DP
    df : Data
    time_index : Time index, used for TREND and TIME_SINCE_FIRST
    prim : primitive to compute
    computed_resources : Map which contain result result for computed primitives
    """

    # Re-Structured
    def mom_vector_1(df):

        if computed_resources['N_ELEMS'] == 0:  # Handling Edge case
            computed_resources['MOM_VECTOR_1'] = np.nan
        else:
            computed_resources['MOM_VECTOR_1'] = computed_resources.get(
                'MOM_VECTOR_1', df - compute_mean(df))

        return computed_resources['MOM_VECTOR_1']

    def mom_vector_2(df):
        computed_resources['MOM_VECTOR_2'] = computed_resources.get(
            'MOM_VECTOR_2', mom_vector_1(df) * mom_vector_1(df))
        return computed_resources['MOM_VECTOR_2']

    def mom_vector_3(df):
        computed_resources['MOM_VECTOR_3'] = computed_resources.get(
            'MOM_VECTOR_3', mom_vector_2(df) * mom_vector_1(df))
        return computed_resources['MOM_VECTOR_3']

    def compute_sum(df):
        computed_resources['SUM'] = computed_resources.get(
            'SUM', np.nansum(df))
        return computed_resources['SUM']

     # Re-restructured

    def compute_mean(df):
        if computed_resources['N_ELEMS'] == 0:  # Handling Edge case
            computed_resources['MEAN'] = np.nan
        else:
            computed_resources['MEAN'] = computed_resources.get(
                'MEAN', (compute_sum(df) / (computed_resources['N_ELEMS'])))
        return computed_resources['MEAN']

    def compute_mode(df):
        if computed_resources['N_ELEMS'] == 0:  # Handling Edge case
            computed_resources['MODE'] = np.nan

        else:
            computed_resources['MODE'] = computed_resources.get(
                'MODE', (statistics.mode(df)))

        return computed_resources['MODE']

    def compute_entropy(df):
        def c_e(df):
            if computed_resources['N_ELEMS_NULL'] <= 1:
                computed_resources['ENTROPY'] = 0
            else:
                unique_count = np.unique(df, return_counts=True)
                computed_resources['ENTROPY'] = scipy.stats.entropy(
                    unique_count[1])
            return computed_resources['ENTROPY']
        computed_resources['ENTROPY'] = computed_resources.get(
            'ENTROPY', c_e(df))
        return computed_resources['ENTROPY']

    def compute_min(df):
        if computed_resources['N_ELEMS'] == 0:  # Handling Edge case
            computed_resources['MIN'] = np.nan
        else:
            computed_resources['MIN'] = computed_resources.get(
                'MIN', np.nanmin(df))
        return computed_resources['MIN']

    def compute_max(df):
        if computed_resources['N_ELEMS'] == 0:  # Handling Edge case
            computed_resources['MAX'] = np.nan
        else:
            computed_resources['MAX'] = computed_resources.get(
                'MAX', np.nanmax(df))
        return computed_resources['MAX']

    def compute_std(df):
        # Population STD
        if computed_resources['N_ELEMS'] == 0:
            computed_resources['STD'] = np.nan
        else:
            computed_resources['STD'] = computed_resources.get('STD', np.sqrt(
                np.sum(mom_vector_2(df)) / (computed_resources['N_ELEMS'] - 1)))
        return computed_resources['STD']

    def compute_skew(df):
        def skew(df):
            m = mom_vector_1(df)
            if compute_num_unique(df) == 1:
                # Special case, When all elements are same, skew is 0
                return 0
            m3 = np.sum(m**3) / computed_resources['N_ELEMS']
            s3 = pow(np.sum(m**2) / (computed_resources['N_ELEMS'] - 1), 1.5)
            comp_skew = (computed_resources['N_ELEMS']**2) / (
                (computed_resources['N_ELEMS'] - 1) * (computed_resources['N_ELEMS'] - 2)) * m3 / s3
            return comp_skew
        try:
            if len(df) <= 2:
                return np.nan
            computed_resources['SKEW'] = computed_resources.get(
                'SKEW', skew(df))
        except Exception as e:
            return np.nan
        return computed_resources['SKEW']

    def compute_last(df):

        if computed_resources['N_ELEMS_NULL'] == 0:  # Edge case
            computed_resources['LAST'] = np.nan
        else:
            computed_resources['LAST'] = computed_resources.get('LAST', df[-1])
        return computed_resources['LAST']

    def compute_first(df):

        if computed_resources['N_ELEMS_NULL'] == 0:  # Edge case
            computed_resources['FIRST'] = np.nan
        else:
            computed_resources['FIRST'] = computed_resources.get(
                'FIRST', df[0])
        return computed_resources['FIRST']

    def compute_trend(df):
        def trend(df):
            X = time_index.astype(float) * 1e-9 / 86400
            y = pd.Series(df)
            lr = LinearRegression(normalize=True)
            lr.fit(X=X.reshape(-1, 1), y=y)
            return lr.coef_
        if computed_resources['N_ELEMS'] <= 2:  # Edge Cases
            computed_resources['TREND'] = np.nan
        else:
            computed_resources['TREND'] = computed_resources.get(
                'TREND', trend(df))
        return computed_resources['TREND']

    def compute_count(df):
        return computed_resources['N_ELEMS']

    # Calculates the time elapsed since the first datetime (in days) till
    # cutoff date
    def compute_time_since_first(df):
        if computed_resources['N_ELEMS'] == 0:  # Edge case
            computed_resources['TIME_SINCE_FIRST'] = np.nan
        else:
            min_date = np.min(time_index)
            computed_resources['TIME_SINCE_FIRST'] = (
                created_time - min_date) / np.timedelta64(1, 'D')
        return computed_resources['TIME_SINCE_FIRST']

    # Calculates the time elapsed since the last datetime (in days) till
    # cutoff date
    def compute_time_since_last(df):
        if computed_resources['N_ELEMS'] == 0:  # Edge case
            computed_resources['TIME_SINCE_LAST'] = np.nan
        else:
            max_date = np.max(time_index)
            computed_resources['TIME_SINCE_LAST'] = (
                created_time - max_date) / np.timedelta64(1, 'D')
        return computed_resources['TIME_SINCE_LAST']

    def compute_num_unique(df):
        u = np.unique(df)
        u = u[np.invert(np.isnan(u))]
        return len(u)

    def compute_day(df):
        return time_index.vals.dt.day

    f_map = {'STD': compute_std,
             'MEAN': compute_mean,
             'SUM': compute_sum,
             'SKEW': compute_skew,
             'ENTROPY': compute_entropy,
             'MIN': compute_min,
             'MAX': compute_max,
             'LAST': compute_last,
             'FIRST': compute_first,
             'TREND': compute_trend,
             'COUNT': compute_count,
             'NUM_UNIQUE': compute_num_unique,
             'TIME_SINCE_LAST': compute_time_since_last,
             'TIME_SINCE_FIRST': compute_time_since_first,
             'MODE': compute_mode
             }

    return f_map[prim](df)


def create_features(df, s, condition, dfa):
    computed_feature_records = {}

    created_time = dfa.created_date.values[0]

    for col, prim_list in s.items():
        time = df.loc[:, 'date'].values
        if col:
            # Feature will be created only when the base columns is present
            if col in df.columns:
                df_col = df.loc[:, col].values
                not_null_mask = np.invert(pd.isnull(df_col))
                df_col_not_null = df_col[not_null_mask]
                time_not_null = time[not_null_mask]
                computed_resources = {
                    'N_ELEMS': len(df_col_not_null),
                    'N_ELEMS_NULL': len(df_col)}
            else:
                warnings.warn(
                    f'The \'{col}\' is not present in data, Depending features will not be created.',
                    stacklevel=2)
                continue
        else:
            # Performing operations on Date column or on entire df, Like COUNT(transactions)
            # Since nulls are removed from time already
            df_col_not_null = time
            time_not_null = time
            computed_resources = {'N_ELEMS': len(df_col_not_null)}

        for prim, key in prim_list:

            if prim in ['LAST', 'FIRST', 'ENTROPY']:
                computed_record = compute_primitive(
                    df_col, time, prim, computed_resources, created_time)
            else:
                computed_record = compute_primitive(
                    df_col_not_null, time_not_null, prim, computed_resources, created_time)

            computed_feature_records.update({key: computed_record})

    return computed_feature_records
