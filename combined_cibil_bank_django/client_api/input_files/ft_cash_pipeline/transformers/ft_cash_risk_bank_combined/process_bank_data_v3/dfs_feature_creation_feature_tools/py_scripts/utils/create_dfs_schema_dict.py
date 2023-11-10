from ..create_schema_buffer import create_schema_buffer
from pickle import load
from constants import *
from custom_exceptions_schema_creation import *
import warnings


def seperate_features_pipeline_wise(features_list):
    pipeline_1_features, pipeline_2_features, pipeline_3_features, pipeline_4_features = [], [], [], []
    if len(features_list) > 0:
        for each in features_list:
            if '__pipeline_1' in each:
                pipeline_1_features.append(each)
            if '__pipeline_2' in each:
                pipeline_2_features.append(each)
            if '__pipeline_3' in each:
                pipeline_3_features.append(each)
            if '__pipeline_4' in each:
                pipeline_4_features.append(each)
    return pipeline_1_features, pipeline_2_features, pipeline_3_features, pipeline_4_features


def load_essentials(
        feature_matrix_stage_1_path,
        feature_matrix_stage_2_path,
        feature_matrix_stage_3_path):
    feature_matrix_stage_1 = load(open(feature_matrix_stage_1_path, 'rb'))
    feature_matrix_stage_2 = load(open(feature_matrix_stage_2_path, 'rb'))
    feature_matrix_stage_3 = load(open(feature_matrix_stage_3_path, 'rb'))
    return feature_matrix_stage_1, feature_matrix_stage_2, feature_matrix_stage_3


def create_definition_mapper(feature_matrix):
    feature_matrix_stage_wise_dict = {}
    for features in feature_matrix:
        feature_matrix_stage_wise_dict[features.get_feature_names()[
            0]] = features
    return feature_matrix_stage_wise_dict


def sort_defintions_to_stages(schema_dict_stage_wise,
                              schema_dict,
                              column_definition_mapper,
                              STAGE_KEYS,
                              SPLIT_VALUE,
                              STAGE_NAME):
    try:
        not_found_features = []
        for different_windows in STAGE_KEYS:
            for subkeys in schema_dict[different_windows].keys():
                for keys, features in schema_dict[values][subkeys].items():
                    for feat_names in features:
                        if feat_names[1].split(SPLIT_VALUE)[
                                0] in column_definition_mapper:
                            if STAGE_NAME != 'stage_3':
                                schema_dict_stage_wise[STAGE_NAME][values[2]].append(
                                    column_definition_mapper[feat_names[1].split(SPLIT_VALUE)[0]])
                            else:
                                schema_dict_stage_wise[STAGE_NAME][values[1]][values[2]].append(
                                    column_definition_mapper[feat_names[1].split(SPLIT_VALUE)[0]])
                        else:
                            not_found_features.append(feat_names)
        return schema_dict_stage_wise, not_found_features
    except Exception as e:
        raise SchemaDictSortingStageError(
            'Sorting Schema Dict Stage Wise Definition has failed, Error: {}'.format(e))


def raise_warning_if_issues(not_found_features, STAGE_ID):
    if len(not_found_features) > 0:
        warnings.warn(
            'Stage {} has {} features for which DFS feature definition is not available\
                . These features will be skipped in feature creation'.format(
                STAGE_ID, len(not_found_features)))
    return


def create_pipeline_basis_schema_dict(pipeline_features,
                                      PIPELINE_STAGE_1_MATRIX_PATH,
                                      PIPELINE_STAGE_2_MATRIX_PATH,
                                      PIPELINE_STAGE_3_MATRIX_PATH):
    schema_dict = {}
    not_found_features_dict = {}
    for features in pipeline_features:
        schema_dict = create_schema_buffer.create_schema_buffer(
            columns, schema_dict)
    feature_matrix_stage_1, feature_matrix_stage_2, feature_matrix_stage_3 = load_essentials(
        PIPELINE_STAGE_1_MATRIX_PATH, PIPELINE_STAGE_2_MATRIX_PATH, PIPELINE_STAGE_3_MATRIX_PATH)

    schema_dict_stage_wise = SCHEMA_DFS_FEATURE_DICT_RAW.copy()
    schema_definition_mapper_stage_1 = create_definition_mapper(
        feature_matrix_stage_1)
    schema_definition_mapper_stage_2 = create_definition_mapper(
        feature_matrix_stage_2)
    schema_definition_mapper_stage_3 = create_definition_mapper(
        feature_matrix_stage_3)
    # For Stage 1
    schema_dict_stage_wise, not_found_features = sort_defintions_to_stages(schema_dict_stage_wise,
                                                                           schema_dict,
                                                                           schema_definition_mapper_stage_1,
                                                                           STAGE_KEYS=STAGE_1_KEYS,
                                                                           SPLIT_VALUE='_features',
                                                                           STAGE_NAME='stage_1')
    not_found_features_dict['stage_1'] = not_found_features
    raise_warning_if_issues(not_found_features, "1")

    # For Stage 2
    schema_dict_stage_wise, not_found_features = sort_defintions_to_stages(schema_dict_stage_wise,
                                                                           schema_dict,
                                                                           schema_definition_mapper_stage_2,
                                                                           STAGE_KEYS=STAGE_2_KEYS,
                                                                           SPLIT_VALUE='',  # Stage 2 features dont have any suffixes
                                                                           STAGE_NAME='stage_2')
    not_found_features_dict['stage_2'] = not_found_features
    raise_warning_if_issues(not_found_features, "2")

    # For Stage 3
    schema_dict_stage_wise, not_found_features = sort_defintions_to_stages(schema_dict_stage_wise,
                                                                           schema_dict,
                                                                           schema_definition_mapper_stage_3,
                                                                           STAGE_KEYS=STAGE_3_KEYS,
                                                                           SPLIT_VALUE='_window_1',
                                                                           STAGE_NAME='stage_3')
    not_found_features_dict['stage_3_w_1'] = not_found_features
    raise_warning_if_issues(not_found_features, "3_window_1")

    schema_dict_stage_wise, not_found_features = sort_defintions_to_stages(schema_dict_stage_wise,
                                                                           schema_dict,
                                                                           schema_definition_mapper_stage_3,
                                                                           STAGE_KEYS=STAGE_3_KEYS,
                                                                           SPLIT_VALUE='_window_30',
                                                                           STAGE_NAME='stage_3')
    not_found_features_dict['stage_3_w_30'] = not_found_features
    raise_warning_if_issues(not_found_features, '3_window_30')

    return schema_dict_stage_wise, not_found_features_dict


def create_dfs_schema_definitions(features_list):
    pipeline_1_features, pipeline_2_features,
    pipeline_3_features, pipeline_4_features = seperate_features_pipeline_wise(
        features_list)
    pipeline_1_not_found_features, pipeline_2_not_found_features, pipeline_3_not_found_features = None, None, None
    pipeline_1_schema = create_pipeline_basis_schema_dict(
        pipeline_1_features=pipeline_1_features,
        PIPELINE_ID=1,
        PIPELINE_STAGE_1_MATRIX_PATH=FEATURE_MATRIX_STAGE_1_PIPELINE_1_PATH,
        PIPELINE_STAGE_2_MATRIX_PATH=FEATURE_MATRIX_STAGE_2_PIPELINE_1_PATH,
        PIPELINE_STAGE_3_MATRIX_PATH=FEATURE_MATRIX_STAGE_3_PIPELINE_1_PATH)
    pipeline_2_schema = create_pipeline_basis_schema_dict(
        pipeline_2_features=pipeline_2_features,
        PIPELINE_ID=2,
        PIPELINE_STAGE_1_MATRIX_PATH=FEATURE_MATRIX_STAGE_1_PIPELINE_2_PATH,
        PIPELINE_STAGE_2_MATRIX_PATH=FEATURE_MATRIX_STAGE_2_PIPELINE_2_PATH,
        PIPELINE_STAGE_3_MATRIX_PATH=FEATURE_MATRIX_STAGE_3_PIPELINE_2_PATH)
    pipeline_3_schema = create_pipeline_basis_schema_dict(
        pipeline_3_features=pipeline_3_features,
        PIPELINE_ID=3,
        PIPELINE_STAGE_1_MATRIX_PATH=FEATURE_MATRIX_STAGE_1_PIPELINE_3_PATH,
        PIPELINE_STAGE_2_MATRIX_PATH=FEATURE_MATRIX_STAGE_2_PIPELINE_3_PATH,
        PIPELINE_STAGE_3_MATRIX_PATH=FEATURE_MATRIX_STAGE_3_PIPELINE_3_PATH)
