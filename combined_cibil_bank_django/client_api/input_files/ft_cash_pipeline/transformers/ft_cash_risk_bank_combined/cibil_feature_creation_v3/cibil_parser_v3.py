from typing import Dict
from typing import Tuple
import copy
import pickle
import multiprocessing as mp
from . import exceptions
from . import lib
from . import constants
from .utils.helper_functions import divide_to_chunks,load_dict_helper
from functools import partial
import pandas as pd

class CibilFeatureCreation:

    def __init__(
            self,
            RENAME_AND_DTYPE_DICT,
            DATE_DICT = {},
            YAML_PATH=None,
            default_date_format = '%Y-%m-%d',
            default_null="(null)",
            SCHEMA_DICT = constants.DEFAULT_SCHEMA_DICT_PATH,
            CLUSTERING_NUMERICAL_INTERESTING_DICT = constants.DEFAULT_CLUSTERING_NUMERICAL_INTERESTING_DICT,
            NUMERICAL_INTERESTING_DICT = constants.DEFAULT_NUMERICAL_INTERESTING_DICT,
            LOAN_TYPE_DICT = constants.DEFAULT_LOAN_TYPE_DICT,
            ACCOUNT_DPD_MAPPING_DICT = constants.DEFAULT_ACCOUNT_DPD_MAPPING,
            ACCOUNT_OWNERSHIP_MAPPING_DICT = constants.DEFAULT_ACCOUNT_OWNERSHIP_MAPPING
            ) -> None:
        """

        This library is used to generate cibil features.

        Arguments:

            RENAME_AND_DTYPE_DICT {dict/ pkl path}  -  [Dictionary of dictionary which contains the columns
                                                 to read/use from each sub section of bureau data and their dtype.
                                                 eg. accounts,enquiry,header,name,score, employment etc ].
                                                 The three accepted dtype are string,numeric and datetime.
                                                
                                                 example:

                                                rename_and_dtype_dict =  {'enquiry': {'Date_of_Enquiry': ('Date_of_Enquiry','datetime'),
                                                    'Enquiring_Member_Short_Name': ('Enquiring_Member_Short_Name','string'),
                                                    'Enquiry_Amount': ('Enquiry_Amount','numeric'),
                                                    'Enquiry_Purpose': ('Enquiry_Purpose','string')},
                                                            
                                                    'header': {
                                                    'enquiryControlNumber': ('enquiryControlNumber','string'),
                                                    'CIBIL_Report_Date': ('CIBIL_Report_Date','datetime')}}  


            DATE_DICTS {dict / pkl path} { default : {} }  - [ Dictionary of dictionary which contains date format
                                                                of the date type columns]
            
            YAML_PATH { YAML /str path} {default : None} - [PATH to the saved YAML file which would be
                                                         used by JsonDataSegregator to traverse
                                                         the JSONs and return df.]
            
            default_date_format {str } {default : '%Y-%m-%d'} - [Default date format to use to read
                                                                datet columns. This is added to 
                                                                DATE_DICTS for each section with the key name 'default']
            
            default_null  {str} { default : "(null)"}  - [Default string to replace null values in data]
            
            

            SCHEMA_DICT {dict/pkl path} - [Dictionary which contains the schema to make features.This should be the output
                                        of CibilFeatureCreation.get_schema_dict(feature_list)]
            
                                        
            CLUSTERING_NUMERICAL_INTERESTING_DICT {dict/pkl path} { default :constants.DEFAULT_CLUSTERING_NUMERICAL_INTERESTING_DICT}
                                                            - [Dictionary which contains numerical columns to be
                                                                converted into categorical buckets for clustering purposes]

                                                                
            NUMERICAL_INTERESTING_DICT {dict/pkl path} { default :constants.DEFAULT_NUMERICAL_INTERESTING_DICT}
                                                            - [Dictionary which contains numerical columns to be
                                                                converted into categorical buckets ]
                            
            LOAN_TYPE_DICT {dict/pkl path} {default: constants.DEFAULT_LOAN_TYPE_DICT}
                                                        - [Dictionary which contains mapping for 
                                                            Account_Type and Enquiry_Purpose]  

            ACCOUNT_DPD_MAPPING_DICT {dict/pkl path} {default: constants.DEFAULT_ACCOUNT_DPD_MAPPING_DICT}
                                                        - [Dictionary which contains mapping for 
                                                            DPD String(Payment History String)]

            ACCOUNT_OWNERSHIP_MAPPING_DICT {dict/pkl path} {default: constants.DEFAULT_ACCOUNT_OWNERSHIP_MAPPING_DICT}
                                                        - [Dictionary which contains mapping for 
                                                            Ownership_Indicator]  
    
            custom_processing_dict {dict} {default : None} - [Custom preprocessing functions to apply to sections]
        
        Returns: A tuple of features and error dictionary
                (pd.DataFrame,{})

        """

        self.RENAME_AND_DTYPE_DICT = self.load_dict(RENAME_AND_DTYPE_DICT)

        self.RENAME_DICT,self.DTYPE_DICT = lib.split_rename_and_dtype_dict(RENAME_AND_DTYPE_DICT)
        self.DATE_DICT = self.load_dict(DATE_DICT)
        self.default_date_format = default_date_format
        self.DATE_DICT = lib.add_default_date(self.DATE_DICT,self.default_date_format)
        self.YAML_PATH = YAML_PATH
        self.default_null = default_null

        self.SCHEMA_DICT = self.load_dict(SCHEMA_DICT)
        self.CLUSTERING_NUMERICAL_INTERESTING_DICT = self.load_dict(CLUSTERING_NUMERICAL_INTERESTING_DICT)
        self.NUMERICAL_INTERESTING_DICT = self.load_dict(NUMERICAL_INTERESTING_DICT)

        self.LOAN_TYPE_DICT = self.load_dict(LOAN_TYPE_DICT)
        self.ACCOUNT_DPD_MAPPING_DICT = self.load_dict(ACCOUNT_DPD_MAPPING_DICT)
        self.ACCOUNT_OWNERSHIP_MAPPING_DICT = self.load_dict(ACCOUNT_OWNERSHIP_MAPPING_DICT)
        
        
        
 
    def load_dict(self, dict_object):
        try:
            if isinstance(dict_object, dict):
                return copy.deepcopy(dict_object)

            elif isinstance(dict_object, str):
                with open(dict_object,'rb') as f:
                    return pickle.load(f)
            
            elif dict_object is None:
                return dict_object
               
        except Exception as e:
            raise exceptions.DictLoadingFailedException(str(e))
        
    

    @staticmethod
    def create_schema_dict(feature_list):
        """static method used to create a schema dict

        Args:
            feature_list (List): List of features for which to create schema dictionary

        Raises:
            exceptions.SchemaDictCreationFailedException: If schema dictionary failed to be created

        Returns:
            schema_dict: Dictionary
        """

        try:

            schema_dict = lib.create_schema_dict(feature_list)
        
        except Exception as e:
            raise exceptions.SchemaDictCreationFailedException(str(e))
        

        return schema_dict

    
    @staticmethod
    def get_default_features(section_list=['account','enquiry','phone',
                                                'address','name','clustering']):

        """
        Return features based on section present in section list

        Raises:
            exceptions.GetDefaultFeaturesFailedException: If get_default_features fails

        Returns:
           selected_features : List of features from cibil selected features 
        """
        
        try:
                
            default_features = load_dict_helper(constants.DEFAULT_FEATURES_LIST_PATH)
            selected_features = []

            if 'account' in section_list:
                selected_features += [feat for feat in default_features if (('account_df' in feat) or ('tl_mom' in feat))]
            

            if 'enquiry' in section_list:
                selected_features += [feat for feat in default_features if ('enquiry_df' in feat)]

            
            if 'phone' in section_list:
                selected_features += [feat for feat in default_features if ('phone_df' in feat)]

            
            if 'address' in section_list:
                selected_features += [feat for feat in default_features if ('address_df' in feat)]

            
            if 'employment' in section_list:
                selected_features += [feat for feat in default_features if ('employment_df' in feat)]
            
            if 'name' in section_list:
                selected_features += [feat for feat in default_features if ('name_df' in feat)]
            

            if 'clustering' in section_list:
                selected_features += [feat for feat in default_features if (('agg_enquiry_clustered_df' in feat) or ('acc_enquiry_clustered_df' in feat))]
        

        except Exception as e:
            raise exceptions.GetDefaultFeaturesFailedException(str(e))
            
        return selected_features
    

    def process_cibil_data_batch(self, data_path, n_jobs=1,
                                 chunk_size=5000,
                                 feature_list=None) -> Tuple[pd.DataFrame,Dict] :
        """
        This function used to create cibil features for batch mode

        Parameters :
            data_path {dict/str path} - [Dictionary of dataframes of sub sections of bureau data or
                                        data path to the csv saved or list of jsons path or list of json]

            n_jobs {int} {default :1} - [Number of cores to use while creating features]

            chunk_size {int} {default :5000} - [Number of unique enquiryControlNumber to be processed with one core]

            feature_list {List} {default: None - [Feature name of which subset should be returned]

        Returns:
            (features {pd.DataFrame},error_dict {Dict}) : A Tuple of DataFrame (consisting all the Features,Cibil_Report_Date
                                                                                and EnquiryControlNumber) and Error dictionary)
        """

        error_dict = {}
        features = pd.DataFrame()
        feature_list_copy = copy.deepcopy(feature_list)
        try:
            if n_jobs == -1:
                n_jobs = mp.cpu_count() - 1

            try:
                #Step 1: Loading All the Data from data_path
                data = lib.load_data(data_path=data_path, 
                                     RENAME_DICTS_=self.RENAME_DICT, 
                                     YAML_PATH=self.YAML_PATH, 
                                     n_jobs=n_jobs)

            except Exception as e:
                # raise e
                error_dict.update(
                    {constants.error_message_str: str(exceptions.LoadingDataFailedException(str(e)))})
                return features,error_dict

            try:
                #Step 2: Renaming Data Columns and Formating Dtypes
                data = lib.rename_and_format_data(dataframe_dict=data,
                                                  RENAME_DICT_=self.RENAME_DICT,
                                                  DTYPE_DICT_=self.DTYPE_DICT,
                                                DATE_DICT_=self.DATE_DICT,
                                                default_null=self.default_null)
            
            except Exception as e:
                # raise e
                error_dict.update(
                    {constants.error_message_str: str(exceptions.RenameFormatDataFailedException(str(e)))})
                return features,error_dict
                

            # STEP 3: Chunk Data For Preprocessing and Feature Creation
            try:
                
                chunked_data = divide_to_chunks(dfd=data,size=chunk_size,n_jobs=n_jobs)

            except Exception as e:
                # raise e
                error_dict.update(
                    {constants.error_message_str: str(exceptions.ChunkingDataFailedException(str(e)))})
                return features,error_dict

            # Step 3: Create Features using mp pool or list comprehension

            try:
                partial_create_features = partial(lib.create_features,
                                                    LOAN_TYPE_DICT=self.LOAN_TYPE_DICT, 
                                                    ACCOUNT_DPD_MAPPING_DICT=self.ACCOUNT_DPD_MAPPING_DICT,
                                                    ACCOUNT_OWNERSHIP_MAPPING_DICT=self.ACCOUNT_OWNERSHIP_MAPPING_DICT,
                                                    NUMERICAL_INTERESTING_DICT=self.NUMERICAL_INTERESTING_DICT,
                                                    CLUSTERING_NUMERICAL_INTERESTING_DICT=self.CLUSTERING_NUMERICAL_INTERESTING_DICT,
                                                    SCHEMA_DICT=self.SCHEMA_DICT)
                
                if n_jobs > 1:
                    pool = mp.Pool(n_jobs)
                    features = pool.map(partial_create_features,chunked_data)
                    pool.close()
                    pool.join()

                 
                else:
                    features = [partial_create_features(data=data_chunk) for data_chunk in chunked_data]

                error_dict_ = [feat[1] for feat in features if feat[1]]
                features = [feat[0] for feat in features]
                features = pd.concat(features,axis=0,ignore_index=True)
                for error in error_dict_:
                    if not 'Error_chunk_create_features' in error_dict:
                        error_dict['Error_chunk_create_features'] = []
                    error_dict['Error_chunk_create_features'].append(error)

                if feature_list is not None:
                    feature_list_copy.append('enquiryControlNumber')
                    feature_list_copy.append('CIBIL_Report_Date')
                    features = features[feature_list_copy]
                return features,error_dict
            except Exception as e:
                # raise e
                error_dict.update(
                    {constants.error_message_str: str(exceptions.FeatureCreationFailedException(str(e)))})
                return features,error_dict
                   
        except Exception as e:
            # raise e
            return features,error_dict.update({constants.error_message_str: str(e)})


    def process_cibil_data_single_json(self, data_path,n_jobs=1,
                                       feature_list=None) -> Tuple[pd.DataFrame,Dict]:
             
            
        """
        This function used to create cibil features for batch mode

        Parameters :
            data_path {dict/str path} - [Dictionary of dataframes of sub sections of bureau data or
                                        data path to the csv saved or list of jsons path or list of json]

            n_jobs {int} {default :1} - [Number of cores to use while creating features]

            feature_list {List} {default: None - [Features name of which subset should be returned]

        Returns:
            (features {pd.DataFrame},error_dict {Dict}) : A Tuple of DataFrame (consisting all the Features,Cibil_Report_Date
                                                                                and EnquiryControlNumber) and Error dictionary)
        """
            
        error_dict = {}
        features = pd.DataFrame()
        feature_list_copy = copy.deepcopy(feature_list)
        try:
            
            try:
                data = lib.load_data(data_path, self.RENAME_DICT, self.YAML_PATH,1)

            except Exception as e:
                error_dict.update(
                    {constants.error_message_str: str(exceptions.LoadingDataFailedException(str(e)))})
                return features,error_dict

            try:
                data = lib.rename_and_format_data(data,self.RENAME_DICT,self.DTYPE_DICT,
                                                    self.DATE_DICT,self.default_null)
            
            except Exception as e:
                error_dict.update(
                    {constants.error_message_str: str(exceptions.RenameFormatDataFailedException(str(e)))})
                return features,error_dict
    

            try:
                
                features,error_dict = lib.create_features_single_json(data=data,
                                            LOAN_TYPE_DICT=self.LOAN_TYPE_DICT, 
                                            ACCOUNT_DPD_MAPPING_DICT=self.ACCOUNT_DPD_MAPPING_DICT,
                                            ACCOUNT_OWNERSHIP_MAPPING_DICT=self.ACCOUNT_OWNERSHIP_MAPPING_DICT,
                                            NUMERICAL_INTERESTING_DICT=self.NUMERICAL_INTERESTING_DICT,
                                            CLUSTERING_NUMERICAL_INTERESTING_DICT=self.CLUSTERING_NUMERICAL_INTERESTING_DICT,
                                            SCHEMA_DICT=self.SCHEMA_DICT)
            
                if feature_list is not None:
                    feature_list_copy.append('enquiryControlNumber')
                    feature_list_copy.append('CIBIL_Report_Date')
                    features = features[feature_list_copy]
                return features,error_dict
            except Exception as e:
                error_dict.update(
                    {constants.error_message_str: str(exceptions.FeatureCreationFailedException(str(e)))})
                return features,error_dict
                
        except Exception as e:
            return features,error_dict.update({constants.error_message_str: str(e)})
    
    def get_processed_cibil_data(self, data_path, n_jobs=1,
                                 chunk_size=5000) -> Tuple[pd.DataFrame,Dict] :
        """
        This function used to create cibil features for batch mode

        Parameters :
            data_path {dict/str path} - [Dictionary of dataframes of sub sections of bureau data or
                                        data path to the csv saved or list of jsons path or list of json]

            n_jobs {int} {default :1} - [Number of cores to use while creating features]

            chunk_size {int} {default :5000} - [Number of unique enquiryControlNumber to be processed with one core]

            feature_list {List} {default: None - [Feature name of which subset should be returned]

        Returns:
            (features {pd.DataFrame},error_dict {Dict}) : A Tuple of DataFrame (consisting all the Features,Cibil_Report_Date
                                                                                and EnquiryControlNumber) and Error dictionary)
        """

        error_dict = {}
        preprocessed_data = None
        try:
            if n_jobs == -1:
                n_jobs = mp.cpu_count() - 1

            try:
                #Step 1: Loading All the Data from data_path
                data = lib.load_data(data_path=data_path, 
                                     RENAME_DICTS_=self.RENAME_DICT, 
                                     YAML_PATH=self.YAML_PATH, 
                                     n_jobs=n_jobs)

            except Exception as e:
                # raise e
                error_dict.update(
                    {constants.error_message_str: str(exceptions.LoadingDataFailedException(str(e)))})
                return preprocessed_data,error_dict

            try:
                #Step 2: Renaming Data Columns and Formating Dtypes
                data = lib.rename_and_format_data(dataframe_dict=data,
                                                  RENAME_DICT_=self.RENAME_DICT,
                                                  DTYPE_DICT_=self.DTYPE_DICT,
                                                DATE_DICT_=self.DATE_DICT,
                                                default_null=self.default_null)
            
            except Exception as e:
                # raise e
                error_dict.update(
                    {constants.error_message_str: str(exceptions.RenameFormatDataFailedException(str(e)))})
                return preprocessed_data,error_dict
                

            # STEP 3: Chunk Data For Preprocessing and Feature Creation
            try:
                
                chunked_data = divide_to_chunks(dfd=data,size=chunk_size,n_jobs=n_jobs)

            except Exception as e:
                # raise e
                error_dict.update(
                    {constants.error_message_str: str(exceptions.ChunkingDataFailedException(str(e)))})
                return preprocessed_data,error_dict

            # Step 3: Create Features using mp pool or list comprehension

            try:
                partial_process_data = partial(lib.preprocess_data_for_feature_creation,
                                                    LOAN_TYPE_DICT=self.LOAN_TYPE_DICT, 
                                                    ACCOUNT_DPD_MAPPING_DICT=self.ACCOUNT_DPD_MAPPING_DICT,
                                                    ACCOUNT_OWNERSHIP_MAPPING_DICT=self.ACCOUNT_OWNERSHIP_MAPPING_DICT,
                                                    NUMERICAL_INTERESTING_DICT=self.NUMERICAL_INTERESTING_DICT,
                                                    CLUSTERING_NUMERICAL_INTERESTING_DICT=self.CLUSTERING_NUMERICAL_INTERESTING_DICT)
                
                if n_jobs > 1:
                    pool = mp.Pool(n_jobs)
                    preprocessed_data = pool.map(partial_process_data,chunked_data)
                    pool.close()
                    pool.join()

                 
                else:
                    preprocessed_data = [partial_process_data(data=data_chunk) for data_chunk in chunked_data]
                
               
                error_dict_ = [data_chunk[1] for data_chunk in preprocessed_data if data_chunk[1]]
                preprocessed_data = [data_chunk[0] for data_chunk in preprocessed_data]

                
                print(len(preprocessed_data))
                keys = preprocessed_data[0].keys()
                preprocessed_data_ = {}
                for key in keys:
                    preprocessed_data_[key] =  pd.concat([data_key[key] for data_key in preprocessed_data],axis=0,ignore_index=True)
                
                for error in error_dict_:
                    if not 'Error_chunk_create_features' in error_dict:
                        error_dict['Error_chunk_create_features'] = []
                    error_dict['Error_chunk_create_features'].append(error)

            
                return preprocessed_data_,error_dict
            except Exception as e:
                # raise e
                error_dict.update(
                    {constants.error_message_str: str(exceptions.FeatureCreationFailedException(str(e)))})
                return preprocessed_data,error_dict
                   
        except Exception as e:
            # raise e
            return preprocessed_data,error_dict.update({constants.error_message_str: str(e)})
