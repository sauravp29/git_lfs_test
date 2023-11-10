
import re
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEXT_CLASSIFICATION_MODEL_PATH = BASE_DIR + \
    '/models_and_artefacts/transaction_type_text_classification_model_v3.pkl'
HOLIDAY_DF_PATH = BASE_DIR + '/models_and_artefacts/holiday_list.csv'
NACH_CLASSIFCATION_MODEL_PATH = BASE_DIR + \
    '/models_and_artefacts/nach_model_v3.pkl'

TFIDF_TXN_TYPE_MODEL_PATH = BASE_DIR + \
    '/models_and_artefacts/tfidf_model_type_of_transaction.pkl'
TFIDF_NACH_MODEL_PATH = BASE_DIR + '/models_and_artefacts/tfidf_model_nach.pkl'
# MODE_OF_PAYMENT_CLASSIFICATION_PATH = 'models_and_artefacts/NACH_MODE_MODELS/LOG_REG_MODE_OF_PAYMENTS.pkl'
COMMON_WORDS_PATH = BASE_DIR + '/models_and_artefacts/common_words_expansion.pkl'

# DEFAULT_PIPELINE_LIST = ['1', '2', '3', '4']

keep_dfs_cols = [
    'application_id',
    'application_created_date',
    'amount',
    'balance',
    'date',
    'type',
    'time_transformed',
    'txn_classified_as',
    'nach_type',
    'is_redundant',
    'datediff_txn',
    'day_type',
    'is_national_holiday',
    'first_7_days',
    'last_7_days',
    'amount_0-1000',
    'amount_1000-5000',
    'amount_5000-10000',
    'amount_10000-20000',
    'amount_20000-50000',
    'amount_50000-',
    'balance_l-0',
    'balance_0-10000',
    'balance_10000-20000',
    'balance_20000-50000',
    'balance_50000-100000',
    'balance_100000-',
    'day_type-type',
    'nach_type-type',
    'is_national_holiday-nach_type',
    'is_national_holiday-type',
    'is_national_holiday-txn_classified_as',
    'nach_type-txn_classified_as',
    'day_type-txn_classified_as',
    'is_redundant-nach_type',
    'is_redundant-txn_classified_as',
    'first_7_days-txn_classified_as',
    'last_7_days-txn_classified_as',
    'amount_0-1000-txn_classified_as',
    'amount_1000-5000-txn_classified_as',
    'amount_5000-10000-txn_classified_as',
    'amount_10000-20000-txn_classified_as',
    'amount_20000-50000-txn_classified_as',
    'amount_50000--txn_classified_as',
    'balance_l-0-txn_classified_as',
    'balance_0-10000-txn_classified_as',
    'balance_10000-20000-txn_classified_as',
    'balance_20000-50000-txn_classified_as',
    'balance_50000-100000-txn_classified_as',
    'balance_100000--txn_classified_as']


REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
URL_REMOVAL_RE = re.compile(r'http\S+')


FRACTION_DAY_DICT = {'nach_type': {300: {},
                                   90: {},
                                   180: {},
                                   30: {}},
                     'txn_classified_as': {300: {},
                                           90: {},
                                           180: {},
                                           30: {}}}

ERROR_MESSAGE_STR = 'Error message'

JSON_EMPTY_STR = 'Json provided as input is invalid or Empty.'
DATA_EMPTY = 'Input provided is empty'
JSON_EXTRACTION_FAILED_STR = 'Some problem occured while extraction data from JSON \n Error : '
TYPE_OF_JSON_FAILED_ERROR = 'Type of json is neither Perfios, Fin360 or Qbera'
NO_DATA_REMAINING_ERROR = 'No Data left after applying custom df subsetting'
CONFIG_DICT_ERROR = 'The CONFIG_DICT must have the following keys\n 1.created_date_format \n 2.transaction_date_format \n 3.create_date_format_unit if format is epoch  \n 4.transaction_date_format_unit if the format is epoch'
PREPROCESS_ERROR = 'Could not prepocess the banking data, this error could also occur because of issue in FILTER DAYS, MAPPER_FUNCTION or df Subset Function: '
TEXTCAT_ERROR = 'error while classifying transaction category of the data: '
NACH_CLASSIFY_ERROR = 'error while classifying nach type for the data: '
DFS_DATA_CREATION_ERROR = 'Could not create data for generating DFS features: '
FRACTION_FEATURE_ERROR = 'error while creating fraction features: '
mappingFileError = 'cannot load mapping file, either mapping_file path is not excel/csv or batch_preprocess is not True'
MULTIPLE_DATES_ERR = 'Multiple created dates found for one application id. Please check the mapping file.'
DATAFRAMERENAMINGEXCEPTION = 'Error in renaming data.'
