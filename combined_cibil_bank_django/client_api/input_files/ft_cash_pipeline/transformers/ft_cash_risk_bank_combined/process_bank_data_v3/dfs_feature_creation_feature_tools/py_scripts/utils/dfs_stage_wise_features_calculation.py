import pandas as pd
from featuretools import Timedelta, calculate_feature_matrix
from multiprocessing import cpu_count
from .dfs_requirement_manager import calculate_temporal_cutoffs, create_stage_2_requirements


def calculate_normal_features(feature_matrix,
                              batch_created_date_mapping,
                              training_window_size_in_days,
                              entity_set,
                              n_jobs=cpu_count(),
                              verbose=True):
    features_df = calculate_feature_matrix(
        features=feature_matrix,
        entityset=entity_set,
        cutoff_time=batch_created_date_mapping,
        training_window=Timedelta(
            training_window_size_in_days,
            "d"),
        n_jobs=n_jobs,
        verbose=verbose)
    features_df = features_df
    return features_df


def calculate_intermediate_features(feature_matrix,
                                    entity_set,
                                    temporal_cutoff,
                                    training_window_size_in_days,
                                    cutoff_time_in_index=True,
                                    n_jobs=cpu_count(),
                                    verbose=True):
    # import pdb;
    # pdb.set_trace()
    features_df = calculate_feature_matrix(
        features=feature_matrix,
        entityset=entity_set,
        cutoff_time=temporal_cutoff,
        training_window=Timedelta(
            training_window_size_in_days,
            "d"),
        n_jobs=n_jobs,
        cutoff_time_in_index=cutoff_time_in_index,
        verbose=verbose,
    )
    features_df = features_df.reset_index(drop=False)
    features_df['date'] = pd.to_datetime(
        features_df['time'], format='%Y-%m-%d')
    features_df = features_df.drop('time', axis=1)
    return features_df
