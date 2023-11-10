import pandas as pd
from .constants import *
from pandas import concat
from multiprocessing import cpu_count
from .dfs_stage_wise_features_calculation import *
from .custom_exceptions import *
from .dfs_requirement_manager import *
import sys
import os
import gc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def get_stagewise_rename_string(STAGE, WINDOW_SIZE, LAST_X_DAYS):
    rename_string = ""
    if STAGE == 1:
        rename_string = rename_string + '_features_{}'.format(LAST_X_DAYS)
    if STAGE == 3:
        rename_string = rename_string + \
            '_window_{}_features_{}'.format(WINDOW_SIZE, LAST_X_DAYS)
    return rename_string


def rename_features(features_dataframe, STAGE, WINDOW_SIZE, LAST_X_DAYS):
    features_columns = features_dataframe.columns.tolist()
    # features_columns.remove('APP_ID')
    rename_dict = {}
    try:
        # TODO: Need to check the Time Column name if it comes and then have to
        # fix it.
        for features in features_columns:
            if features == "APP_ID" or features == 'time':
                continue
            else:
                rename_string = get_stagewise_rename_string(
                    STAGE, WINDOW_SIZE, LAST_X_DAYS)
                rename_dict[features] = features + rename_string
        features_dataframe = features_dataframe.rename(rename_dict, axis=1)
        #features_dataframe = features_dataframe.drop('APP_ID', axis = 1)
        return features_dataframe
    except Exception as e:
        raise FeatureRenameError(
            "Renaming stage Wise Features Failed, Error: {}".format(e))


def create_stage_1_features(entity_set,
                            SCHEMA_DEFINITIONS,
                            batch_created_date_mapping,
                            n_jobs=cpu_count() - 1,
                            verbose=True):
    df_30, df_90, df_300 = None, None, None
    try:
        for keys in SCHEMA_DEFINITIONS.keys():
            if (keys == 30) and (len(SCHEMA_DEFINITIONS[keys]) > 0):
                df_30 = calculate_normal_features(
                    SCHEMA_DEFINITIONS[30],
                    batch_created_date_mapping,
                    training_window_size_in_days=30,
                    entity_set=entity_set,
                    n_jobs=n_jobs,
                    verbose=verbose)
                # print(df_30.index)
                df_30 = rename_features(
                    df_30, STAGE=1, WINDOW_SIZE=None, LAST_X_DAYS=30)
            if (keys == 90) and (len(SCHEMA_DEFINITIONS[keys]) > 0):
                df_90 = calculate_normal_features(
                    SCHEMA_DEFINITIONS[90],
                    batch_created_date_mapping,
                    training_window_size_in_days=90,
                    entity_set=entity_set,
                    n_jobs=n_jobs,
                    verbose=verbose)
                df_90 = rename_features(
                    df_90, STAGE=1, WINDOW_SIZE=None, LAST_X_DAYS=90)
            if (keys == 300) and (len(SCHEMA_DEFINITIONS[keys]) > 0):
                df_300 = calculate_normal_features(
                    SCHEMA_DEFINITIONS[300],
                    batch_created_date_mapping,
                    training_window_size_in_days=300,
                    entity_set=entity_set,
                    n_jobs=n_jobs,
                    verbose=verbose)
                df_300 = rename_features(
                    df_300, STAGE=1, WINDOW_SIZE=None, LAST_X_DAYS=300)
        return df_30, df_90, df_300
    except Exception as e:
        raise Stage1FeatureCreationError(
            'Stage 1 Feature Creation Failed. Error {}'.format(e))


def create_stage_2_features(entity_set,
                            SCHEMA_DEFINITIONS,
                            batch_created_date_mapping,
                            n_jobs=cpu_count() - 1,
                            verbose=True):
    try:
        df_window_1, df_window_30 = None, None
        dfa_instance_ids, dfa_instance_time, es_first_txn_time = create_stage_2_requirements(
            entity_set, batch_created_date_mapping)
        for keys in SCHEMA_DEFINITIONS.keys():
            if (keys == 1) and (len(SCHEMA_DEFINITIONS[keys]) > 0):
                temporal_cutoff_window_1 = calculate_temporal_cutoffs(
                    dfa_instance_ids, dfa_instance_time, es_first_txn_time, cutoff_time=1)
                # print(SCHEMA_DEFINITIONS[1])
                df_window_1 = calculate_intermediate_features(
                    feature_matrix=SCHEMA_DEFINITIONS[1],
                    entity_set=entity_set,
                    temporal_cutoff=temporal_cutoff_window_1,
                    training_window_size_in_days=1,
                    n_jobs=n_jobs,
                    verbose=verbose)

            if (keys == 30) and (len(SCHEMA_DEFINITIONS[keys]) > 0):
                temporal_cutoff_window_30 = calculate_temporal_cutoffs(
                    dfa_instance_ids, dfa_instance_time, es_first_txn_time, cutoff_time=30)
                df_window_30 = calculate_intermediate_features(
                    feature_matrix=SCHEMA_DEFINITIONS[30],
                    entity_set=entity_set,
                    temporal_cutoff=temporal_cutoff_window_30,
                    training_window_size_in_days=30,
                    n_jobs=n_jobs,
                    verbose=verbose)
        return df_window_1, df_window_30
    except Exception as e:
        raise Stage2FeatureCreationError(
            'Stage 2 Feature Creation Failed. Error {}'.format(e))


def create_stage_3_window_1_features(entity_set,
                                     SCHEMA_DEFINITIONS,
                                     batch_created_date_mapping,
                                     n_jobs=cpu_count() - 1,
                                     verbose=True):
    df_window_1_feat_30, df_window_1_feat_90, df_window_1_feat_300 = None, None, None
    try:
        for window in SCHEMA_DEFINITIONS.keys():
            for last_days in SCHEMA_DEFINITIONS[window].keys():
                if (window == 1) and (last_days == 30) and (
                        len(SCHEMA_DEFINITIONS[window][last_days]) > 0):
                    df_window_1_feat_30 = calculate_normal_features(
                        SCHEMA_DEFINITIONS[window][30],
                        batch_created_date_mapping,
                        training_window_size_in_days=30,
                        entity_set=entity_set,
                        n_jobs=n_jobs,
                        verbose=verbose)
                    df_window_1_feat_30 = rename_features(
                        df_window_1_feat_30, STAGE=3, WINDOW_SIZE=window, LAST_X_DAYS=last_days)
                if (window == 1) and (last_days == 90) and (
                        len(SCHEMA_DEFINITIONS[window][last_days]) > 0):
                    df_window_1_feat_90 = calculate_normal_features(
                        SCHEMA_DEFINITIONS[window][90],
                        batch_created_date_mapping,
                        training_window_size_in_days=90,
                        entity_set=entity_set,
                        n_jobs=n_jobs,
                        verbose=verbose)
                    df_window_1_feat_90 = rename_features(
                        df_window_1_feat_90, STAGE=3, WINDOW_SIZE=window, LAST_X_DAYS=last_days)
                if (window == 1) and (last_days == 300) and (
                        len(SCHEMA_DEFINITIONS[window][last_days]) > 0):
                    df_window_1_feat_300 = calculate_normal_features(
                        SCHEMA_DEFINITIONS[window][300],
                        batch_created_date_mapping,
                        training_window_size_in_days=300,
                        entity_set=entity_set,
                        n_jobs=n_jobs,
                        verbose=verbose)
                    df_window_1_feat_300 = rename_features(
                        df_window_1_feat_300, STAGE=3, WINDOW_SIZE=window, LAST_X_DAYS=last_days)

        return df_window_1_feat_30, df_window_1_feat_90, df_window_1_feat_300
    except Exception as e:
        raise Stage3FeatureCreationError(
            'Stage 3 Feature Creation Failed. Error {}'.format(e))


def create_stage_3_window_30_features(entity_set,
                                      SCHEMA_DEFINITIONS,
                                      batch_created_date_mapping,
                                      n_jobs=cpu_count() - 1,
                                      verbose=True):
    df_window_30_feat_300 = None
    try:
        for window in SCHEMA_DEFINITIONS.keys():
            for last_days in SCHEMA_DEFINITIONS[window].keys():
                if (window == 30) and (last_days == 300) and (
                        len(SCHEMA_DEFINITIONS[window][last_days]) > 0):
                    df_window_30_feat_300 = calculate_normal_features(
                        SCHEMA_DEFINITIONS[window][300],
                        batch_created_date_mapping,
                        training_window_size_in_days=300,
                        entity_set=entity_set,
                        n_jobs=n_jobs,
                        verbose=verbose)
                    df_window_30_feat_300 = rename_features(
                        df_window_30_feat_300, STAGE=3, WINDOW_SIZE=window, LAST_X_DAYS=last_days)

        return df_window_30_feat_300
    except Exception as e:
        raise Stage3FeatureCreationError(
            'Stage 3 Feature Creation Failed. Error {}'.format(e))


def segregate_schema_definitions(SCHEMA_DEFINITIONS):
    stage_1_schema, stage_2_schema, stage_3_schema = None, None, None
    try:
        if SCHEMA_DEFINITIONS.get('stage_1', None):
            stage_1_schema = SCHEMA_DEFINITIONS['stage_1']
        if SCHEMA_DEFINITIONS.get('stage_2', None):
            stage_2_schema = SCHEMA_DEFINITIONS['stage_2']
        if SCHEMA_DEFINITIONS.get('stage_3', None):
            stage_3_schema = SCHEMA_DEFINITIONS['stage_3']
        return stage_1_schema, stage_2_schema, stage_3_schema
    except Exception as e:
        raise SchemaDictSegregationError(
            'Schema Dict Unpack Error. Error {}'.format(e))


def concat_list_of_dataframes(list_of_dataframe):
    try:
        final_list_to_concat = []
        for dataframe in list_of_dataframe:
            if isinstance(dataframe, pd.DataFrame):
                final_list_to_concat.append(dataframe)
        final_features_df = concat(final_list_to_concat, axis=1)
        return final_features_df
    except Exception as e:
        raise FeatureConcatenationError(
            'Final Feature Concatenation has failed. Error {}'.format(e))


def convert_cols_numeric(df):
    try:
        id_dt_cols = ['APP_ID', 'created_date', 'date', 'time_transformed']
        num_cols = [col for col in df.columns if df[col].dtype !=
                    'O' and col not in id_dt_cols and df[col].nunique() > 2]
        df[num_cols] = df[num_cols].astype('float32')
        return df
    except Exception as e:
        raise FloatConversionError(
            'Converting Numerical features to Float failed. Error {}'.format(e))


def create_dfs_features_featuretools(
        batch_dataframe,
        batch_created_date_mapping,
        SCHEMA_DEFINITIONS,
        pre_required_dfs_cols,
        n_jobs=cpu_count() - 1,
        verbose=True):
    stage_1_schema, stage_2_schema, stage_3_schema = segregate_schema_definitions(
        SCHEMA_DEFINITIONS['SCHEMA'])
    STAGE_NAME = 'stage_1'
    print('Starting stage 1 feature creation')
    req_cols = [
        'APP_ID',
        'created_date',
        'date',
        'amount',
        'balance',
        'type',
        'time_transformed',
        'datediff_txn']
    stage_1_batch_dataframe = batch_dataframe[list(
        set(req_cols + pre_required_dfs_cols))]
    stage_1_batch_dataframe = convert_cols_numeric(stage_1_batch_dataframe)
    entity_set = intialise_dfs(
        stage_1_batch_dataframe, STAGE_NAME=STAGE_NAME)
    df_30, df_90, df_300 = create_stage_1_features(entity_set=entity_set,
                                                   SCHEMA_DEFINITIONS=stage_1_schema,
                                                   batch_created_date_mapping=batch_created_date_mapping,
                                                   n_jobs=n_jobs, verbose=verbose)

    print('Stage 1 features created')

    del entity_set

    STAGE_NAME = 'stage_2'
    # print(batch_dataframe.columns)
    print('Starting stage 2 feature creation')
    stage_2_batch_dataframe = batch_dataframe.drop(
        columns=STAGE_2_COLS_TO_REMOVE)
    #stage_2_batch_dataframe = convert_col_numeric(stage_2_batch_dataframe)
    entity_set = intialise_dfs(
        batch_dataframe, STAGE_NAME=STAGE_NAME)
    df_window_1, df_window_30 = create_stage_2_features(entity_set=entity_set,
                                                        SCHEMA_DEFINITIONS=stage_2_schema,
                                                        batch_created_date_mapping=batch_created_date_mapping,
                                                        n_jobs=n_jobs,
                                                        verbose=verbose)

    print('Stage 2 features created')
    del entity_set
    gc.collect()
    # import pdb;
    # pdb.set_trace()

    STAGE_NAME = 'stage_3'
    print('Starting stage 3 feature creation')
    entity_set = intialise_dfs(
        df_window_1, STAGE_NAME=STAGE_NAME)
    df_window_1_feat_30, df_window_1_feat_90, df_window_1_feat_300 = create_stage_3_window_1_features(
        entity_set=entity_set, SCHEMA_DEFINITIONS=stage_3_schema, batch_created_date_mapping=batch_created_date_mapping, n_jobs=n_jobs, verbose=verbose)

    # Reintialising entity set basis the stage 2 dfs for stage 3 feature
    # creation
    del entity_set
    gc.collect()
    STAGE_NAME = 'stage_3'
    entity_set = intialise_dfs(
        df_window_30, STAGE_NAME=STAGE_NAME)
    df_window_30_feat_300 = create_stage_3_window_30_features(
        entity_set=entity_set,
        SCHEMA_DEFINITIONS=stage_3_schema,
        batch_created_date_mapping=batch_created_date_mapping,
        n_jobs=n_jobs,
        verbose=verbose)
    print('Stage 3 features created')
    del entity_set
    gc.collect()
    # print(df_30.index)
    final_features = concat_list_of_dataframes([df_30,
                                                df_90,
                                                df_300,
                                                df_window_1_feat_30,
                                                df_window_1_feat_90,
                                                df_window_1_feat_300,
                                                df_window_30_feat_300])

    final_features = final_features.reset_index(drop=False)
    return final_features
