import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_MATRIX_STAGE_1_PATH = BASE_DIR + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_1_final_10765.pkl'
FEATURE_MATRIX_STAGE_2_W_1_PATH = BASE_DIR + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_2_updated.pkl'
FEATURE_MATRIX_STAGE_2_W_30_PATH = BASE_DIR + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_2_w_30.pkl'
FEATURE_MATRIX_STAGE_3_W_1_PATH = BASE_DIR + \
    '/dfs_feature_creation_feature_tools/feature_definitions/feature_matrix_stage_3_window_1_final.pkl'
FEATURE_MATRIX_STAGE_3_W_30_PATH = BASE_DIR + \
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

STAGE_1_COLS_TO_REMOVE = ['amount_50000--txn_classified_as',
                          'is_national_holiday-nach_type',
                          'time_transformed',
                          'amount_0-1000',
                          'created_date',
                          'amount_20000-50000-txn_classified_as',
                          'amount_20000-50000',
                          'amount_50000-',
                          'description',
                          'last_7_days',
                          'is_redundant',
                          'balance_10000-20000',
                          'balance_100000--txn_classified_as',
                          'balance_100000-',
                          'balance_0-10000',
                          'balance_50000-100000',
                          'amount_10000-20000-txn_classified_as',
                          'balance_20000-50000',
                          'amount_1000-5000',
                          'first_7_days',
                          'balance_l-0',
                          'day_type',
                          'amount_10000-20000',
                          'balance_50000-100000-txn_classified_as',
                          'filtered_description',
                          'amount_5000-10000',
                          'is_national_holiday']

STAGE_2_COLS_TO_REMOVE = ['time_transformed',
                          'amount_0-1000',
                          'created_date',
                          'amount_20000-50000',
                          'amount_50000-',
                          'description',
                          'last_7_days',
                          'is_redundant',
                          'balance_10000-20000',
                          'ind',
                          'balance_100000-',
                          'balance_0-10000',
                          'balance_50000-100000',
                          'balance_20000-50000',
                          'amount_1000-5000',
                          'first_7_days',
                          'balance_l-0',
                          'day_type',
                          'amount_10000-20000',
                          'filtered_description',
                          'amount_5000-10000',
                          'is_national_holiday']
