from functools import partial
import pandas as pd
import pickle
import yaml
from functools import reduce
import operator
import logging
import multiprocessing as mp
from .banking_json_to_df import parse_monsoon, rename_dataframe_using_yaml
from . import CONSTANT
from .feature_creation_dfs.test_dfs_manager import create_dfs_features
from .exceptions import *
from . import lib
import math
from .dfs_feature_creation_feature_tools.py_scripts.schema_dict_creation.create_dfs_schema_dict import create_dfs_schema_definitions
from .dfs_feature_creation_feature_tools.py_scripts.utils.run_dfs_pipeline import create_dfs_features_featuretools
from .utils.create_schema_buffer import create_schema_buffer
# from processing_raw_data.json_mapper_new.src.json_segregator_v2 import JsonDataSegregator
from .utils.get_required_columns_pre_dfs import get_required_columns
import time
import gc
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class BankingParser():

    def __init__(
        self,
        YAML_PATH,
        FILTER_MAX_DAYS=300,
        MAPPER_FORMATTER_FUNCTION_DICT=None,
        df_subset_custom_function=None,
        ADD_CUSTOM_FEATURE_PRE_LIST=None,
        FEATURE_LIST_OR_PATH=BASE_DIR +
        '/models_and_artefacts/1722_final_selected_features.pkl',
        PRE_DFS_ADDED_COLUMN=None,
        MAPPING_FILE=None,
        ROUND_TOL=8,
    ):

        self.YAML_PATH = YAML_PATH
        self.FILTER_MAX_DAYS = FILTER_MAX_DAYS
#         self.jds_object = JsonDataSegregator(
#             YAML_PATH=self.YAML_PATH, OUTPUT_TYPE=1)
        # self.TYPE_OF_JSON= TYPE_OF_JSON
        self.ROUND_TOL = ROUND_TOL
        self.TEXT_CLASSIFICATION_MODEL = self.load_text_classification_model(
            CONSTANT.TEXT_CLASSIFICATION_MODEL_PATH)

        # self.PIPELINE_1_FEAT, self.PIPELINE_2_FEAT, self.PIPELINE_3_FEAT, self.PIPELINE_4_FEAT = self.get_pipeline_wise_feat(self.FEATURE_LIST)
        self.HOLIDAY_DF = self.load_holiday_df(CONSTANT.HOLIDAY_DF_PATH)
        self.YAML_FILE = self.load_yaml(self.YAML_PATH)
        self.CONFIG_DICT = self.YAML_FILE['default_config_dict']
        self.NACH_CLASSIFICATION_MODEL = self.load_classification_model(
            CONSTANT.NACH_CLASSIFCATION_MODEL_PATH)

        self.TFIDF_TXN_TYPE_MODEL = self.load_classification_model(
            CONSTANT.TFIDF_TXN_TYPE_MODEL_PATH
        )

        self.TFIDF_NACH_MODEL = self.load_classification_model(
            CONSTANT.TFIDF_NACH_MODEL_PATH
        )
        # Required for Bank data DFS
        self.MAPPER_FORMATTER_FUNCTION_DICT = MAPPER_FORMATTER_FUNCTION_DICT
        self.df_subset_custom_function = df_subset_custom_function
        self.ADD_CUSTOM_FEATURE_PRE_LIST = ADD_CUSTOM_FEATURE_PRE_LIST
        self.PRE_DFS_ADDED_COLUMN = PRE_DFS_ADDED_COLUMN
        # self.DFS_FEATURE = [i for i in self.FEATURE_LIST if 'features' in i]
        if (MAPPING_FILE is not None):
            self.MAPPING_FILE = self.load_mapping_file(MAPPING_FILE)
        self.FEATURE_LIST = self.load_feature_list(FEATURE_LIST_OR_PATH)
        self.pre_dfs_required_cols = get_required_columns(self.FEATURE_LIST)
        self.FRACTION_FEATURES = [
            i for i in self.FEATURE_LIST if 'frac__' in i]
        self.DFS_FEATURE = list(
            set(self.FEATURE_LIST) - set(self.FRACTION_FEATURES))
        self.SCHEMA_DICT = self.create_new_schema_dict(self.DFS_FEATURE)
        self.FEATURETOOLS_SCHEMA_DICT = self.create_schema_definitions_featuretools(
            self.DFS_FEATURE)
        self.COMMON_WORDS_EXPANSION = self.load_pickled_object(
            CONSTANT.COMMON_WORDS_PATH)

    def load_yaml(self, _YAML_PATH_):
        '''
        This function will load the provided yaml
        Args:
            _YAML_PATH_ : path of yaml
        Output:
            json: unpacked yaml
        '''
        try:
            with open(_YAML_PATH_, 'r') as file:
                yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        except Exception as e:
            logging.exception(e)
            # make exception
            raise Exception('Please provide YAML path Or correct Yaml File.')
            return None
        return yaml_data

    def load_feature_list(self, FEATURE_LIST_OR_PATH):
        try:
            if isinstance(FEATURE_LIST_OR_PATH, list):
                return FEATURE_LIST_OR_PATH
            else:
                return pickle.load(open(FEATURE_LIST_OR_PATH, 'rb'))
        except Exception as e:
            raise e

    def load_mapping_file(self, mapping_file_path):
        try:
            if isinstance(mapping_file_path, str):
                if 'xlsx' in mapping_file_path:
                    mapping_file = pd.read_excel(mapping_file_path)

                elif 'csv' in mapping_file_path:
                    mapping_file = pd.read_csv(mapping_file_path)

                return mapping_file

            else:
                return mapping_file_path

        except Exception as e:
            raise MappingFileLoadingFailedException(CONSTANT.mappingFileError)

    def create_new_schema_dict(self, FEATURES_LIST):
        """
        Return the schema dict for given featues
        FEATURES_LIST - Is the list of final features names based in schema dict will be created
        Returns : schema_dict, Pass it to BankStatementsParser SCHEMA_DICT_PATH_OR_DICT to create the features specified in FEATURES_LIST
        """
        try:
            schema_dict = {}
            for columns in FEATURES_LIST:
                schema_dict = create_schema_buffer(
                    columns, schema_dict)
            return schema_dict
        except Exception as e:
            raise e

    def create_schema_definitions_featuretools(self, FEATURES_LIST):
        """
        Return the schema dict for given featues
        FEATURES_LIST - Is the list of final features names based in schema dict will be created
        Returns : schema_dict, Pass it to BankStatementsParser SCHEMA_DICT_PATH_OR_DICT to create the features specified in FEATURES_LIST
        """
        try:
            schema_dict = create_dfs_schema_definitions(FEATURES_LIST)
            return schema_dict
        except Exception as e:
            raise e

    def load_classification_model(self, MODE_CLASSIFICATION_MODEL_PATH):
        try:
            return pickle.load(
                open(MODE_CLASSIFICATION_MODEL_PATH, 'rb'))
        except Exception as e:
            raise ClassficationModelLoadingFailedException(e)

    def load_pickled_object(self, OBJECT_PATH):
        try:
            return pickle.load(
                open(OBJECT_PATH, 'rb'))
        except Exception as e:
            raise ClassficationModelLoadingFailedException(e)

    def load_holiday_df(self, HOLIDAY_DF_PATH):
        try:
            if isinstance(HOLIDAY_DF_PATH, pd.DataFrame):
                return HOLIDAY_DF_PATH
            else:
                holiday_df = pd.read_csv(HOLIDAY_DF_PATH)
                return holiday_df
        except Exception as e:
            raise HolidayDFLoadingFailedException(e)

    def load_text_classification_model(self, TEXT_CLASSIFICATION_MODEL_PATH):
        try:
            return pickle.load(
                open(TEXT_CLASSIFICATION_MODEL_PATH, 'rb'))
        except Exception as e:
            raise TextClassificationModelLoadingFailedException(e)

    def generate_error_dict(self, data_dict, error):
        data_dict['error'] = error
        return data_dict

    def remove_errorneous_dict(self, data):
        while True:
            try:
                data.remove({})
            except BaseException:
                break
        return data

    def process_bank_data_json(self,
                               bank_json_raw,
                               n_jobs=None,
                               batch_prepoc=False
                               ):

        error_dict = {}
        # STEP 1 Get the data from JSON
        try:
            # import pdb;
            # pdb.set_trace()
            if not batch_prepoc:
                # st = time.time()
                bank_json = self.jds_object.unpack_json(bank_json_raw)
                # print(time.time() - st)
                if 'Error Message' in bank_json:
                    error_dict.update(bank_json)
                    return error_dict

                self.APP_ID = bank_json['informative_dict']['App_id']

                self.created_date = bank_json['informative_dict']['created_date']

                data = parse_monsoon(
                    bank_json,
                    self.APP_ID,
                    self.created_date,
                    self.CONFIG_DICT['deduce_type_from_amount'])

            else:
                final_df_dict = []
                for raw_json in bank_json_raw:
                    try:
                        bank_json = self.jds_object.unpack_json(raw_json)
                        if 'Error Message' in bank_json:
                            error_dict.update(bank_json)
                            return error_dict
                        self.APP_ID = bank_json['informative_dict']['App_id']
                        self.created_date = bank_json['informative_dict']['created_date']
                        data = parse_monsoon(
                            bank_json,
                            self.APP_ID,
                            self.created_date,
                            self.CONFIG_DICT['deduce_type_from_amount'])
                        final_df_dict.extend(data.to_dict(orient='records'))
                    except Exception as e:
                        print(e)
                data = pd.DataFrame(final_df_dict)

        except Exception as e:
            error_dict.update({CONSTANT.ERROR_MESSAGE_STR: JsonExtractionDataFailedException(
                CONSTANT.JSON_EXTRACTION_FAILED_STR + str(e)).message})
            return error_dict
        feature_dict = self.get_features_bank_data(data, batch_prepoc)
        return feature_dict

    def process_bank_data_df(self, i, g1, g2):
        error_dict = {}
        try:
            try:
                # print(i)
                dfi = g1.get_group(i)
                dfa = g2.get_group(i)
            except BaseException:
                logging.warning(
                    f'Application id does not exist in both the files. Skipping. {i}')
                return {}
                # return error_dict.update({CONSTANT.ERROR_MESSAGE_STR:
                # 'Application id does not exist in both the files.
                # Skipping.'})
            dfi = dfi.drop_duplicates()
            dfa = dfa.drop_duplicates()
            app_id = self.YAML_FILE['transactions']['application_id']
            created_date = self.YAML_FILE['transactions']['created_date']
            if dfa.shape[0] == 1:
                dfi['application_created_date'] = dfa[created_date].iloc[0]
            else:
                error_dict.update({CONSTANT.ERROR_MESSAGE_STR: MultiplecreatedDateException(
                    CONSTANT.MULTIPLE_DATES_ERR + str(e)).message})
                return error_dict
            data = rename_dataframe_using_yaml(
                dfi, self.YAML_FILE, i, deduce_type=self.CONFIG_DICT['deduce_type_from_amount'])
            feature_dict = self.get_features_bank_data(data)
            return feature_dict
        except Exception as e:
            # print(CONSTANT.ERROR_MESSAGE_STR, e)
            error_dict.update({CONSTANT.ERROR_MESSAGE_STR: str(e)})
            # print(error_dict)
            return error_dict

    def get_features_bank_data(self, data, batch_prepoc=False, n_jobs=None):
        error_dict = {}
        if data.shape[0] <= 0:
            error_dict.update({CONSTANT.ERROR_MESSAGE_STR: DataEmptyException(
                CONSTANT.DATA_EMPTY + str(e)).message})
            return error_dict
        try:

            # created_date = pd.to_datetime(created_date,format = self.CONFIG_DICT["created_date_format"]).strftime('%Y-%m-%d')

            error_dict['APP_ID'] = data['application_id']
            banking_df = lib.preprocess_banking_data(
                data, self.MAPPER_FORMATTER_FUNCTION_DICT,
                self.CONFIG_DICT, self.df_subset_custom_function,
                self.FILTER_MAX_DAYS, batch_prepoc)
            created_date = data['application_created_date'].iloc[0]
            final_feature_dict = {
                'APP_ID': data['application_id'].iloc[0],
                'created_date': created_date}
        except RawBankingPreprocessFailedException as e:
            error_dict.update({CONSTANT.ERROR_MESSAGE_STR: e.message})
            return error_dict
        if not banking_df.empty:
            # STEP 3: Run text classification model to classify into 18
            # different txn classes
            try:
                classified_txn_data_df = lib.text_classification(
                    banking_df,
                    self.TEXT_CLASSIFICATION_MODEL,
                    self.TFIDF_TXN_TYPE_MODEL,
                    self.NACH_CLASSIFICATION_MODEL,
                    self.TFIDF_NACH_MODEL,
                    self.HOLIDAY_DF,
                    self.COMMON_WORDS_EXPANSION)
                # print(self.HOL)
            except TextClassificationFailedException as e:
                error_dict.update({CONSTANT.ERROR_MESSAGE_STR: e.message})
                return error_dict
            try:
                txn_all_classes = self.TEXT_CLASSIFICATION_MODEL.classes_.tolist()
                nach_all_classes = self.NACH_CLASSIFICATION_MODEL.classes_.tolist()
                # print(classified_txn_data_df.columns)
                # return classified_txn_data_df
                final_pre_dfs_data, dfa, mapper_dict_keys = lib.create_pre_dfs_data(
                    classified_txn_data_df, self.HOLIDAY_DF, self.FEATURE_LIST, self.PRE_DFS_ADDED_COLUMN, BATCH_PREPROCESS=batch_prepoc)
                dfa = dfa.rename(
                    columns={
                        'application_created_date': 'created_date'})
                # final_pre_dfs_data  = final_pre_dfs_data[CONSTANT.keep_dfs_cols]
                if isinstance(self.ADD_CUSTOM_FEATURE_PRE_LIST, list):
                    if len(self.ADD_CUSTOM_FEATURE_PRE_LIST) == 0:
                        return final_pre_dfs_data
                    elif len(self.ADD_CUSTOM_FEATURE_PRE_LIST) > 0:
                        for custom_func in self.ADD_CUSTOM_FEATURE_PRE_LIST:
                            final_pre_dfs_data = custom_func(
                                final_pre_dfs_data)
                if batch_prepoc:
                    fraction_df = final_pre_dfs_data.groupby('application_id').apply(
                        lambda x: lib.create_fraction_features(
                            x,
                            self.FRACTION_FEATURES,
                            txn_all_classes,
                            nach_all_classes)).to_frame('dicts').reset_index()
                    fraction_feature_df = pd.DataFrame(
                        list(fraction_df['dicts']))
                    fraction_feature_df['APP_ID'] = fraction_df['application_id']
                    del fraction_df
                    return final_pre_dfs_data, dfa, fraction_feature_df
                # return final_pre_dfs_data
            except AdditionalColumnCreationFailedException as e:
                error_dict.update({CONSTANT.ERROR_MESSAGE_STR: e.message})
                return error_dict
        try:
            fraction_feature_dict = lib.create_fraction_features(
                final_pre_dfs_data, self.FRACTION_FEATURES, txn_all_classes, nach_all_classes)
            final_feature_dict.update(fraction_feature_dict)

        except FractionFeatureCreationFaield as e:
            error_dict.update({CONSTANT.ERROR_MESSAGE_STR: e.message})
            return error_dict

        dfs_features = create_dfs_features(
            dfi=final_pre_dfs_data,
            dfa=dfa,
            schema_dict=self.SCHEMA_DICT,
            mapper_dict_keys=mapper_dict_keys,
            internal_mp=False,
            n_jobs=n_jobs)

        dfs_feat_dict = dfs_features.to_dict('records')[0]
        final_feature_dict.update(dfs_feat_dict)
        final_feature_dict = pd.DataFrame([final_feature_dict])
        final_feature_dict = final_feature_dict.round(self.ROUND_TOL)
        return final_feature_dict

    def process_bank_data_batch(self,
                                bank_json_list,
                                n_jobs=mp.cpu_count() - 1):
        '''
        This function will convert a batch of raw json to processed json
        Args:
            bank_json_list : loaded raw json list
            n_jobs: Number of processors (By default it will use all available process - 1). Pass multiprocessing.cpu_count() to use all cores.
        Output:
            list of processed_json

        '''
        pool_process = mp.Pool(processes=n_jobs)
        if isinstance(bank_json_list, list):
            results = pool_process.map(
                self.process_bank_data_json, bank_json_list)
        else:
            if self.MAPPING_FILE is None:
                raise MappingFileNotFoundError('Mapping File not Found')
            app_id = self.YAML_FILE['transactions']['application_id']
            created_date = self.YAML_FILE['transactions']['created_date']
            self.MAPPING_FILE = self.MAPPING_FILE[[app_id, created_date]]
            g1 = bank_json_list.groupby(app_id)
            g2 = self.MAPPING_FILE.groupby(app_id)
            bank_json_list = bank_json_list[app_id].unique().tolist()
            # print(bank_json_list.columns)
           # print(self.MAPPING_FILE.columns)
           # return
            # print(type(bank_json_list), len(bank_json_list))
            results = pool_process.map(
                partial(
                    self.process_bank_data_df,
                    g1=g1,
                    g2=g2),
                bank_json_list)
        pool_process.close()
        pool_process.join()
#         bank_features_df_temp = [file for file in results if (isinstance(file,pd.Series) or isinstance(file,pd.DataFrame))]
#         if len(bank_features_df_temp)>0:
#             bank_features_df_temp = pd.concat(bank_features_df_temp,axis=0,ignore_index=True)
#             bank_features_df_temp = bank_features_df_temp.reset_index(drop=True)
#         else:
#             bank_features_df_temp = pd.DataFrame()
#         error = [file for file in results if not (isinstance(file,pd.Series) or isinstance(file,pd.DataFrame))]
#         results = self.remove_errorneous_dict(results)
#         results = pd.concat(results)
#         results = results.reset_index(drop=True)
        return results#bank_features_df_temp,error

        # load yaml
        # rename using YAML,
        # make a new fucntion in banking json which takes in df
        # create a new function in parser_copy which takes in list of app_id, g1 object of dfi and g2 object of mapper
        # multi proc with multiple arguments
        # get_group and check mappinbg should have only one row.
        # iloc[0] add to tahe dfi
        # run the banking parser function in jsontodf
        # use a single fucntion after data in process raw data
        # check memory consumption

    def process_bank_data_feature_tools(self,
                                        bank_json_list,
                                        batch_size=5000,
                                        n_jobs=mp.cpu_count() - 1):
        # ret_final_features = []
        final_dfa = []
        final_dfs_features = []
        final_frac_features = []
        batch_div = math.ceil(len(bank_json_list) / batch_size)
        print(f'{batch_div} Batches created')
        for i in range(batch_div):
            start = i * batch_size
            end = start + batch_size
            if i == batch_div - 1:
                end = len(bank_json_list)
            out_process_bank = self.process_bank_data_json(
                bank_json_list[start: end], batch_prepoc=True, n_jobs=n_jobs,)
            if not isinstance(out_process_bank, tuple):
                return out_process_bank
            ret_final_features, ret_dfa, frac_feats = out_process_bank[
                0], out_process_bank[1], out_process_bank[2]
            final_frac_features.extend(frac_feats.to_dict('r'))
            del frac_feats
            del out_process_bank
            # ret_final_features.extend(final_features.to_dict('r'))
            # ret_dfa.extend(dfa.to_dict('r'))
            print(f'Batch {i + 1} Preprocess done')
            # return ret_final_features, ret_dfa
            ret_final_features = pd.DataFrame(ret_final_features)
            ret_dfa = pd.DataFrame(
                ret_dfa).drop_duplicates().reset_index(drop=True)
            ret_final_features['datediff_txn'] = ret_final_features['datediff_txn'].astype(
                'int32')
            # ret_dfa = pd.DataFrame(ret_dfa).drop_duplicates().reset_index(drop = True)
            ret_dfa = ret_dfa.rename(
                columns={
                    'application_id': 'instance_id',
                    'created_date': 'time'})
            ret_final_features = ret_final_features.rename(
                columns={
                    'application_id': 'APP_ID',
                    'application_created_date': 'created_date'})
            ret_final_features['APP_ID'] = ret_final_features['APP_ID'].astype(
                str)
            dfs_features_tools = create_dfs_features_featuretools(
                ret_final_features,
                ret_dfa,
                self.FEATURETOOLS_SCHEMA_DICT,
                self.pre_dfs_required_cols,
                n_jobs=n_jobs)
            del ret_final_features
            final_dfs_features.extend(dfs_features_tools.to_dict('r'))
            del dfs_features_tools
            final_dfa.extend(ret_dfa.to_dict('r'))
            del ret_dfa
            gc.collect()
            print('\n')
        # final_frac_features = pd.DataFrame.from_dict(final_frac_features)
        final_frac_features = pd.DataFrame(final_frac_features)
        final_dfs_features = pd.DataFrame(final_dfs_features)
        final_dfa = pd.DataFrame(final_dfa)
        final_dfa = final_dfa.rename(
            columns={
                'instance_id': 'APP_ID',
                'time': 'created_date'})
        # print(final_dfa.columns, final_dfa.shape)
        final_dfs_features = final_dfs_features.merge(
            final_dfa, on='APP_ID', how='inner')
        final_dfs_features = final_dfs_features.merge(
            final_frac_features, on='APP_ID', how='inner')
        final_dfs_features = final_dfs_features.round(self.ROUND_TOL)
        # dfs_features = create_dfs_features(ret_final_features, ret_dfa, self.FEATURETOOLS_SCHEMA_DICT)
        # return final_dfs_features
        return final_dfs_features
