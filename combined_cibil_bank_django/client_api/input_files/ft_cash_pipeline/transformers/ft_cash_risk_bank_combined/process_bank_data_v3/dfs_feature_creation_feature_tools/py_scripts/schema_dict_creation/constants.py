import os
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rel_path = str(Path(BASE_DIR).parents[2])
FEATURE_MATRIX_STAGE_1_PATH = rel_path + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_1_final_10765.pkl'
FEATURE_MATRIX_STAGE_2_W_1_PATH = rel_path + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_2_updated.pkl'
FEATURE_MATRIX_STAGE_2_W_30_PATH = rel_path + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_2_w_30.pkl'
FEATURE_MATRIX_STAGE_3_W_1_PATH = rel_path + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_3_window_1_final.pkl'
FEATURE_MATRIX_STAGE_3_W_30_PATH = rel_path + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_3_window_30_actual.pkl'


SCHEMA_DFS_FEATURE_DICT_RAW = {
    'stage_1': {
        30: [],
        90: [],
        300: []
    },
    'stage_2': {
        1: [],
        30: []
    },
    'stage_3': {
        1: {30: [],
            90: [],
            300: []},
        30: {300: []}
    }
}
# These keys naming conventions are from the create schema buffer script
# and it will be required to subset the features
STAGE_1_KEYS = [('_features_300', None, 300, 1),
                ('_features_30', None, 30, 1),
                ('_features_90', None, 90, 1)]

STAGE_2_KEYS = [('_intermediate_1_features_300', 1, 300, 1),
                ('_intermediate_30_features_300', 30, 300, 1)]

STAGE_3_W_1_KEYS = [('_window_1_features_90', 1, 90, 2),
                    ('_window_1_features_30', 1, 30, 2),
                    ('_window_1_features_300', 1, 300, 2),
                    ]

STAGE_3_W_30_KEYS = [('_window_30_features_300', 30, 300, 2)]
