from copy import deepcopy
import warnings
from .custom_exceptions_schema_creation import *
from .constants import *
from pickle import load
from .create_schema_buffer import create_schema_buffer
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def seperate_features_pipeline_wise(features_list):
    #pipeline_1_features, pipeline_2_features, pipeline_3_features, pipeline_4_features, not_found_features = [], [], [], [], []

    if len(features_list) > 0:
        not_found_features = []
        for each in features_list:

            s11 = ('_features_30' in each)
            s12 = ('_features_90' in each)
            s13 = ('_features_300' in each)

            s21 = ('_window_1_features_30' in each)
            s22 = ('_window_1_features_90' in each)
            s23 = ('_window_1_features_300' in each)

            s31 = ('_window_30_features_300' in each)

            if not any([s11, s12, s13, s21, s22, s23, s31]):
                not_found_features.append(each)

    return not_found_features


def create_definition_mapper(
        feature_matrix_stage_1_path,
        feature_matrix_stage_2_1_path,
        feature_matrix_stage_2_30_path,
        feature_matrix_stage_3_1_path,
        feature_matrix_stage_3_30_path):
    feature_matrix_stage_1 = load(open(feature_matrix_stage_1_path, 'rb'))
    feature_matrix_stage_2_w1 = load(open(feature_matrix_stage_2_1_path, 'rb'))
    feature_matrix_stage_2_w30 = load(
        open(feature_matrix_stage_2_30_path, 'rb'))
    feature_matrix_stage_3_w1 = load(open(feature_matrix_stage_3_1_path, 'rb'))
    feature_matrix_stage_3_w30 = load(
        open(feature_matrix_stage_3_30_path, 'rb'))
    stage_1_smapper = create_definition_mapper_utils([feature_matrix_stage_1])
    stage_2_smapper = create_definition_mapper_utils(
        [feature_matrix_stage_2_w1, feature_matrix_stage_2_w30])
    stage_3_smapper = create_definition_mapper_utils(
        [feature_matrix_stage_3_w1, feature_matrix_stage_3_w30])
    return stage_1_smapper, stage_2_smapper, stage_3_smapper


def create_definition_mapper_utils(feature_matrix):
    feature_matrix_stage_wise_dict = {}
    for all_mats in feature_matrix:
        for features in all_mats:
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
            # By Aniket N.
            if different_windows in schema_dict:
                for subkeys in schema_dict[different_windows].keys():
                    for keys, features in schema_dict[different_windows][subkeys].items(
                    ):
                        for feat_names in features:
                            if STAGE_NAME == 'stage_1' and feat_names[1].split(
                                    SPLIT_VALUE)[0] in column_definition_mapper:
                                schema_dict_stage_wise[STAGE_NAME][different_windows[2]].append(
                                    column_definition_mapper[feat_names[1].split(SPLIT_VALUE)[0]])
                            elif STAGE_NAME == 'stage_2' and feat_names[1] in column_definition_mapper:
                                schema_dict_stage_wise[STAGE_NAME][different_windows[1]].append(
                                    column_definition_mapper[feat_names[1]])
                            elif STAGE_NAME == 'stage_3' and feat_names[1].split(SPLIT_VALUE)[0] in column_definition_mapper:
                                schema_dict_stage_wise[STAGE_NAME][different_windows[1]][different_windows[2]].append(
                                    column_definition_mapper[feat_names[1].split(SPLIT_VALUE)[0]])
                            else:
                                not_found_features.append(feat_names)
        return schema_dict_stage_wise, not_found_features
    except Exception as e:
        raise SchemaDictSortingStageError(
            'Sorting Schema Dict Stage Wise Definition has failed, Error: {}'.format(e))


def raise_warning_if_issues(not_found_features, STAGE_ID):
    if len(not_found_features) > 0:
        if STAGE_ID is None:
            warnings.warn(
                'Pipeline or stage name could not be recognised for {} features.\n These features will be skipped in feature creation {}'.format(
                    len(not_found_features), ' '.join(not_found_features)))

        else:
            warnings.warn(
                'Stage {} has {} features for which DFS feature definition is not available. These features will be skipped in feature creation {}'.format(
                    STAGE_ID,
                    len(not_found_features),
                    ' '.join([i[1] for i in not_found_features])))
    return


def create_pipeline_basis_schema_dict(pipeline_features,
                                      STAGE_1_MATRIX_PATH,
                                      STAGE_2_MATRIX_PATH_W_1,
                                      STAGE_2_MATRIX_PATH_W_30,
                                      STAGE_3_MATRIX_PATH_W_1,
                                      STAGE_3_MATRIX_PATH_W_30):

    schema_dict = {}
    not_found_features_dict = {}
    # By Aniket N. To fix the issue of empty list being sent
    # print("pipeline_features",pipeline_features)
    if len(pipeline_features) != 0:
        for features in pipeline_features:
            features = features.replace('windows', 'window')
            schema_dict = create_schema_buffer(
                features, schema_dict)
#             print("features",features, "\nschema_dict",schema_dict)
#             print('*'*30)
        # feature_matrix_stage_1, feature_matrix_stage_2, feature_matrix_stage_3 = load_essentials(PIPELINE_STAGE_1_MATRIX_PATH,
            #  PIPELINE_STAGE_2_MATRIX_PATH,
            #  PIPELINE_STAGE_3_MATRIX_PATH)
#         print("feature_matrix_stage_2\n", feature_matrix_stage_2)
        schema_dict_stage_wise = deepcopy(SCHEMA_DFS_FEATURE_DICT_RAW)
        schema_definition_mapper_stage_1, schema_definition_mapper_stage_2, schema_definition_mapper_stage_3 = create_definition_mapper(
            STAGE_1_MATRIX_PATH, STAGE_2_MATRIX_PATH_W_1, STAGE_2_MATRIX_PATH_W_30, STAGE_3_MATRIX_PATH_W_1, STAGE_3_MATRIX_PATH_W_30)
#         print("\nschema_definition_mapper_stage_2\n",schema_definition_mapper_stage_2)
        # For Stage 1
#         print(PIPELINE_ID, ": Stage 1 schema_dict_stage_wise", schema_dict_stage_wise)

        schema_dict_stage_wise, not_found_features = sort_defintions_to_stages(schema_dict_stage_wise,
                                                                               schema_dict,
                                                                               schema_definition_mapper_stage_1,
                                                                               STAGE_KEYS=STAGE_1_KEYS,
                                                                               SPLIT_VALUE='_features',
                                                                               STAGE_NAME='stage_1')
        not_found_features_dict['stage_1'] = not_found_features
        raise_warning_if_issues(not_found_features, "1")

        # For Stage 2
#         print("\nschema_dict_stage_wise\n", schema_dict_stage_wise)
        schema_dict_stage_wise, not_found_features = sort_defintions_to_stages(schema_dict_stage_wise,
                                                                               schema_dict,
                                                                               schema_definition_mapper_stage_2,
                                                                               STAGE_KEYS=STAGE_2_KEYS,
                                                                               SPLIT_VALUE='####',
                                                                               # Stage
                                                                               # 2
                                                                               # features
                                                                               # dont
                                                                               # have
                                                                               # any
                                                                               # suffixes
                                                                               # so
                                                                               # adding
                                                                               # a
                                                                               # random
                                                                               # split
                                                                               # value
                                                                               # (Need
                                                                               # a
                                                                               # smarter
                                                                               # logic
                                                                               # to
                                                                               # fix
                                                                               # this)
                                                                               STAGE_NAME='stage_2')

        not_found_features_dict['stage_2'] = not_found_features
        raise_warning_if_issues(not_found_features, "2")

        # For Stage 3
        schema_dict_stage_wise, not_found_features = sort_defintions_to_stages(schema_dict_stage_wise,
                                                                               schema_dict,
                                                                               schema_definition_mapper_stage_3,
                                                                               STAGE_KEYS=STAGE_3_W_1_KEYS,
                                                                               SPLIT_VALUE='_window_1',
                                                                               STAGE_NAME='stage_3')
        not_found_features_dict['stage_3_w_1'] = not_found_features
        raise_warning_if_issues(not_found_features, "3_window_1")

        schema_dict_stage_wise, not_found_features = sort_defintions_to_stages(schema_dict_stage_wise,
                                                                               schema_dict,
                                                                               schema_definition_mapper_stage_3,
                                                                               STAGE_KEYS=STAGE_3_W_30_KEYS,
                                                                               SPLIT_VALUE='_window_30',
                                                                               STAGE_NAME='stage_3')
        not_found_features_dict['stage_3_w_30'] = not_found_features
        raise_warning_if_issues(not_found_features, '3_window_30')

    else:
        schema_dict_stage_wise = {}
    return schema_dict_stage_wise, not_found_features_dict


def create_final_schema_dict(final_features,
                             not_found_features):
    return{"SCHEMA": final_features,
           'NOT_FOUND_FEATURES': not_found_features
           }


def create_dfs_schema_definitions(features_list):
    #    try:
    if True:
        not_found_features = seperate_features_pipeline_wise(
            features_list)
        raise_warning_if_issues(not_found_features, None)

        #pipeline_1_not_found_features, pipeline_2_not_found_features, pipeline_3_not_found_features = None, None, None
#        print("pipeline_1_features\n", pipeline_1_features)
#        print("pipeline_2_features\n", pipeline_2_features)
#         print("pipeline_3_features\n", pipeline_3_features)
#         print("pipeline_4_features\n", pipeline_4_features)
#         pdb.set_trace()
#         import pdb;
#         pdb.set_trace()
        final_schema, not_found_features = create_pipeline_basis_schema_dict(
            pipeline_features=features_list,
            STAGE_1_MATRIX_PATH=FEATURE_MATRIX_STAGE_1_PATH,
            STAGE_2_MATRIX_PATH_W_1=FEATURE_MATRIX_STAGE_2_W_1_PATH,
            STAGE_2_MATRIX_PATH_W_30=FEATURE_MATRIX_STAGE_2_W_30_PATH,
            STAGE_3_MATRIX_PATH_W_1=FEATURE_MATRIX_STAGE_3_W_1_PATH,
            STAGE_3_MATRIX_PATH_W_30=FEATURE_MATRIX_STAGE_3_W_30_PATH)

        return create_final_schema_dict(
            final_features=final_schema,
            not_found_features=not_found_features)
#     except Exception as e:
#         raise SchemaDictFeatureResolvingError(
#             'Sorting Schema Dict and Feature Resolution has failed, Error: {}'.format(e))
