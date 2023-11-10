
import copy
import numpy as np
import math

import pandas as pd
from .utils.loading_data import load_data_from_data_dict#, load_data_from_data_list
from .utils.process_header import process_header_df
from .utils.process_enquiry import process_enquiry_df
from .utils.process_account import process_account_df
# from .utils.process_score import process_score_df
from .utils.process_name import process_name_df
# from .utils.process_identification import process_id_df
from .utils.process_phone import process_phone_df
from .utils.process_address import process_address_df
from .utils.process_employment import process_employment_df

from . import constants

from .utils.preprocess_and_clean_dataframe_dict import reconcile_datasets_rd, format_and_rename_cols

from .clustering.clustering_main import generate_clustering_features_data

from .dfs.dfs_config import run_dfs

from .utils.helper_functions import create_schema_dict_from_feature_list,make_numerical_interesting_vars

from .old_dfs.processing_tl_mom import process_tl_mom_df
from .exceptions import (RawDataPreprocessFailedException,ClusteringDFSFeatureCreationFailedException,
                         DFSFeatureCreationFailedException)



def split_rename_and_dtype_dict(RENAME_AND_DTYPE_DICT):
    RENAME_DICT = {}
    DTYPE_DICT = {}

    for dataset,sub_rename_and_dtype_dict in RENAME_AND_DTYPE_DICT.items():
        RENAME_DICT[dataset] = {}
        DTYPE_DICT[dataset] = {}
        for column_name,column_value_list in sub_rename_and_dtype_dict.items():
            RENAME_DICT[dataset][column_name] = column_value_list[0]
            DTYPE_DICT[dataset][column_name] = column_value_list[1]
            
        if 'enquiryControlNumber' not in RENAME_DICT[dataset]:
            RENAME_DICT[dataset]['enquiryControlNumber'] = 'enquiryControlNumber'
            DTYPE_DICT[dataset]['enquiryControlNumber'] = 'string'
            

    print('RENAME_AND_DTYPE_DICT splitted')
    return RENAME_DICT,DTYPE_DICT


def add_default_date(DATE_DICT,default_date_format):
    
    for dateset in constants.DATASETS:
        if dateset not in DATE_DICT:
            DATE_DICT[dateset] = {}
        
        DATE_DICT[dateset]['default'] = default_date_format
    
    print('added default date format to DATE_DICT')
    
    return DATE_DICT


def load_data(data_path, RENAME_DICTS_, YAML_PATH, n_jobs):
    RENAME_DICTS = copy.deepcopy(RENAME_DICTS_)

    if isinstance(data_path, dict):
        return load_data_from_data_dict(data_path, RENAME_DICTS)

    elif isinstance(data_path, list):
        return load_data_from_data_list(
            data_path, RENAME_DICTS, YAML_PATH, n_jobs)
    else:
        raise ValueError(f'{type(data_path)} not understood. Please provide dict or list of jsons')


def rename_and_format_data(dataframe_dict,
                           RENAME_DICT_,
                         DTYPE_DICT_,
                        DATE_DICT_,
                        default_null):
    
    RENAME_DICT = copy.deepcopy(RENAME_DICT_)
    DTYPE_DICT = copy.deepcopy(DTYPE_DICT_)
    DATE_DICT = copy.deepcopy(DATE_DICT_)
     #replace default null in dataframe_dict
    for name in dataframe_dict:
        dataframe_dict[name] = dataframe_dict[name].replace(
            [default_null,None], [np.nan,np.nan])

    dataframe_dict = reconcile_datasets_rd(
        dataframe_dict, RENAME_DICT)

    for name in dataframe_dict:
        dataframe_dict[name] = format_and_rename_cols(
            dataframe_dict[name],
            DTYPE_DICT[name],
            RENAME_DICT[name],
            DATE_DICT[name])
    
    return dataframe_dict
    



def preprocess_data(
        dataframe_dict,
        LOAN_TYPE_DICT_,
        ACCOUNT_DPD_MAPPING_DICT_,
        ACCOUNT_OWNERSHIP_MAPPING_DICT_,
        NUMERICAL_INTERESTING_DICT_,
        SCHEMA_DICT_
        ):


    LOAN_TYPE_DICT = copy.deepcopy(LOAN_TYPE_DICT_)
    ACCOUNT_DPD_MAPPING_DICT = copy.deepcopy(ACCOUNT_DPD_MAPPING_DICT_)
    ACCOUNT_OWNERSHIP_MAPPING_DICT  = copy.deepcopy(ACCOUNT_OWNERSHIP_MAPPING_DICT_)
    NUMERICAL_INTERESTING_DICT = copy.deepcopy(NUMERICAL_INTERESTING_DICT_)
    SCHEMA_DICT = copy.deepcopy(SCHEMA_DICT_)
    
    

    if 'header' in dataframe_dict:
        dataframe_dict['header'] = process_header_df(dataframe_dict['header'])

    
    if 'enquiry' in dataframe_dict and 'enquiry' in SCHEMA_DICT:
        dataframe_dict['enquiry'] = process_enquiry_df(
            dataframe_dict['enquiry'], dataframe_dict['header'],
            LOAN_TYPE_DICT,SCHEMA_DICT['enquiry'])
        
        if 'enquiry' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['enquiry'] = make_numerical_interesting_vars(dataframe_dict['enquiry'],
                                                                        NUMERICAL_INTERESTING_DICT['enquiry'])
     
        common_enq_cols = SCHEMA_DICT['enquiry'][2].intersection(set(dataframe_dict['enquiry'].columns))
        dataframe_dict['enquiry'] = dataframe_dict['enquiry'][list(common_enq_cols)]

       
    if 'account' in dataframe_dict:
        if 'account' in SCHEMA_DICT or 'tl_mom' in SCHEMA_DICT or 'stage_2_dfs_features' in SCHEMA_DICT:

            dataframe_dict['account'] = process_account_df(
                dataframe_dict['account'], dataframe_dict['header'],LOAN_TYPE_DICT,
                ACCOUNT_DPD_MAPPING_DICT,ACCOUNT_OWNERSHIP_MAPPING_DICT)


            if 'account' in NUMERICAL_INTERESTING_DICT:
                dataframe_dict['account'] = make_numerical_interesting_vars(dataframe_dict['account'],
                                                                            NUMERICAL_INTERESTING_DICT['account'])
            
            #TODO: Optimize process_tl_mom_df
            if 'tl_mom' in SCHEMA_DICT or 'stage_2_dfs_features' in SCHEMA_DICT:
                dataframe_dict['tl_mom'] = process_tl_mom_df(dataframe_dict['account'],SCHEMA_DICT['tl_mom'][2])
                # common_tl_mom_cols = SCHEMA_DICT['tl_mom'][2].intersection(set(dataframe_dict['tl_mom'].columns))
                # dataframe_dict['tl_mom'] = dataframe_dict['tl_mom'][list(common_tl_mom_cols)]
        
            common_acc_cols = SCHEMA_DICT['account'][2].intersection(set(dataframe_dict['account'].columns))
            dataframe_dict['account'] = dataframe_dict['account'][list(common_acc_cols)]

    # if 'score' in dataframe_dict:
    #     dataframe_dict['score'], = process_score_df(
    #         dataframe_dict['score'], dataframe_dict['header'])
    
    
    if 'name' in dataframe_dict and 'name' in SCHEMA_DICT:
        dataframe_dict['name'] = process_name_df(dataframe_dict['name'],dataframe_dict['header'])

        if 'name' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['name'] = make_numerical_interesting_vars(dataframe_dict['name'],
                                                                        NUMERICAL_INTERESTING_DICT['name'])

    # if 'identification' in dataframe_dict:
    #     dataframe_dict['identification'], RENAME_DICTS['identification'] = process_id_df(
    #         dataframe_dict['identification'], dataframe_dict['header'], RENAME_DICTS['identification'])
    #     cols_for_interesting_values = []

 
    if 'phone' in dataframe_dict and 'phone' in SCHEMA_DICT:
        dataframe_dict['phone'] = process_phone_df(
            dataframe_dict['phone'])

        if 'phone' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['phone'] = make_numerical_interesting_vars(dataframe_dict['phone'],
                                                                        NUMERICAL_INTERESTING_DICT['phone'])

    if 'address' in dataframe_dict and 'address' in SCHEMA_DICT:
        dataframe_dict['address'] = process_address_df(
            dataframe_dict['address'], dataframe_dict['header'])

        if 'address' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['address'] = make_numerical_interesting_vars(dataframe_dict['address'],
                                                                        NUMERICAL_INTERESTING_DICT['address'])
        

    if 'employment' in dataframe_dict and 'employment' in SCHEMA_DICT:
        dataframe_dict['employment'] = process_employment_df(
            dataframe_dict['employment'], dataframe_dict['header'])
        
        if 'employment' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['employment'] = make_numerical_interesting_vars(dataframe_dict['employment'],
                                                                        NUMERICAL_INTERESTING_DICT['employment'])
    
    return dataframe_dict


def create_clustering_features_data(dataframe_dict,CLUSTERING_NUMERICAL_INTERESTING_DICT_):

    CLUSTERING_NUMERICAL_INTERESTING_DICT = copy.deepcopy(CLUSTERING_NUMERICAL_INTERESTING_DICT_)
    acc_cluster_matched_df,agg_enq_clustered_df = generate_clustering_features_data(dataframe_dict['header'],
                                                       dataframe_dict['account'],
                                                       dataframe_dict['enquiry'],
                                                       num_interesting_value_dict=CLUSTERING_NUMERICAL_INTERESTING_DICT)

    


    return acc_cluster_matched_df,agg_enq_clustered_df



def create_dfs_features(dataframe_dict,schema_dict_):

    schema_dict = copy.deepcopy(schema_dict_)

    date_col_dict = {

        'account' : 'Date_Opened_Disbursed',
        'enquiry' : 'Date_of_Enquiry',
        'name' : None,
        'address' : 'Date_Reported',
        'employment' : 'Date_Reported',
        'phone' : None,
        'acc_enquiry_clustered' : 'Date_Opened_Disbursed',
        'agg_enquiry_clustered' : 'min_DOE',
        'tl_mom' : 'Pmt_Date',

    }

    unique_ids = dataframe_dict['header']['enquiryControlNumber'].unique()


    features = []

    #creating stage 2 dfs features

    if 'stage_2_dfs_features' in schema_dict:
        account_schema_dict = schema_dict['stage_2_dfs_features'][0]
        tl_mom_schema_dict = schema_dict['stage_2_dfs_features'][1]
        ratio_dict = schema_dict['stage_2_dfs_features'][2]

        print(f'creating stage 2 dfs features')
        # tl mom features
        tl_mom_temp_features = []
        for training_window,time_sub_schema_dict in tl_mom_schema_dict.items():
            temp_features = run_dfs(df=dataframe_dict['tl_mom'],
                                    unique_ids=dataframe_dict['tl_mom']['tl_u_id'].unique(),
                                    schema_dict=time_sub_schema_dict,
                                    date_col=date_col_dict['tl_mom'],
                                    training_window=training_window,
                                    groupby_col='tl_u_id')
           
            tl_mom_temp_features.append(temp_features)

        tl_mom_temp_features  = pd.concat(tl_mom_temp_features,axis=1,join='outer',copy=False)

        dataframe_dict['account'] = dataframe_dict['account'].merge(tl_mom_temp_features,right_index=True,
                                                                    left_on='tl_u_id',how='left',suffixes=('','_y'))
        
        
        for training_window,time_sub_schema_dict in account_schema_dict.items():
            temp_features = run_dfs(df=dataframe_dict['account'],
                                    unique_ids=unique_ids,
                                    schema_dict=time_sub_schema_dict,
                                    date_col=date_col_dict['account'],
                                    training_window=training_window)
            features.append(temp_features)
            
        
        schema_dict['stage_2_dfs_features'] = [{},ratio_dict]
        
   
    for dataset,subset_schema_dict in schema_dict.items():
        print(f'creating dfs features for {dataset}')
        for training_window,time_sub_schema_dict in subset_schema_dict[0].items():
            temp_features = run_dfs(df=dataframe_dict[dataset],
                                    unique_ids=unique_ids,
                                    schema_dict=time_sub_schema_dict,
                                    date_col=date_col_dict[dataset],
                                    training_window=training_window)
            features.append(temp_features)
    

    features  = pd.concat(features,axis=1,join='outer',copy=False)
    print(features.shape)

    for dataset,subset_schema_dict in schema_dict.items():
        print(f'creating ratio features for {dataset}')
        for name,(ratio_col_1,ratio_col_2) in subset_schema_dict[1].items():
            if ratio_col_1 in features.columns and ratio_col_2 in features.columns:
                name = 'frac_'+ratio_col_1+f'___{ratio_col_2}'
                features[name] = features[ratio_col_1]/features[ratio_col_2]
                features[name] = features[name].replace([-np.inf,np.inf],np.nan)
            else:
                print(f'{ratio_col_1} or {ratio_col_2} is not present in df columns')
        
    features['enquiryControlNumber'] = features.index
    return features


def create_dfs_features_single_json(dataframe_dict,schema_dict_):

    schema_dict = copy.deepcopy(schema_dict_)

    date_col_dict = {

        'account' : 'Date_Opened_Disbursed',
        'enquiry' : 'Date_of_Enquiry',
        'name' : None,
        'address' : 'Date_Reported',
        'employment' : 'Date_Reported',
        'phone' : None,
        'acc_enquiry_clustered' : 'Date_Opened_Disbursed',
        'agg_enquiry_clustered' : 'min_DOE',
        'tl_mom' : 'Pmt_Date',

    }

    unique_ids = [dataframe_dict['header']['enquiryControlNumber'].iloc[0]]


    features = []

    #creating stage 2 dfs features

    if 'stage_2_dfs_features' in schema_dict:
        account_schema_dict = schema_dict['stage_2_dfs_features'][0]
        tl_mom_schema_dict = schema_dict['stage_2_dfs_features'][1]
        ratio_dict = schema_dict['stage_2_dfs_features'][2]

        print(f'creating stage 2 dfs features')
        # tl mom features
        tl_mom_temp_features = []
        for training_window,time_sub_schema_dict in tl_mom_schema_dict.items():
            temp_features = run_dfs(df=dataframe_dict['tl_mom'],
                                    unique_ids=dataframe_dict['tl_mom']['tl_u_id'].unique(),
                                    schema_dict=time_sub_schema_dict,
                                    date_col=date_col_dict['tl_mom'],
                                    training_window=training_window,
                                    groupby_col='tl_u_id')
           
            tl_mom_temp_features.append(temp_features)

        tl_mom_temp_features  = pd.concat(tl_mom_temp_features,axis=1,join='outer',copy=False)

        dataframe_dict['account'] = dataframe_dict['account'].merge(tl_mom_temp_features,right_index=True,
                                                                    left_on='tl_u_id',how='left',suffixes=('','_y'))
        
        
        for training_window,time_sub_schema_dict in account_schema_dict.items():
            temp_features = run_dfs(df=dataframe_dict['account'],
                                    unique_ids=unique_ids,
                                    schema_dict=time_sub_schema_dict,
                                    date_col=date_col_dict['account'],
                                    training_window=training_window)
            features.append(temp_features)
            
        
        schema_dict['stage_2_dfs_features'] = ({},ratio_dict)
        
   
    for dataset,subset_schema_dict in schema_dict.items():
        print(f'creating dfs features for {dataset}')
        for training_window,time_sub_schema_dict in subset_schema_dict[0].items():
            temp_features = run_dfs(df=dataframe_dict[dataset],
                                    unique_ids=unique_ids,
                                    schema_dict=time_sub_schema_dict,
                                    date_col=date_col_dict[dataset],
                                    training_window=training_window)
            features.append(temp_features)
    

    features  = pd.concat(features,axis=1,join='outer',copy=False)
    features['enquiryControlNumber'] = features.index
    print(features.shape)

    for dataset,subset_schema_dict in schema_dict.items():
        print(f'creating ratio features for {dataset}')
        for name,(ratio_col_1,ratio_col_2) in subset_schema_dict[1].items():
            if ratio_col_1 in features.columns and ratio_col_2 in features.columns:
                name = 'frac_'+ratio_col_1+f'___{ratio_col_2}'
                features[name] = features[ratio_col_1]/features[ratio_col_2]
                features[name] = features[name].replace([-np.inf,np.inf],np.nan)
            else:
                print(f'{ratio_col_1} or {ratio_col_2} is not present in df columns')

    return features


def create_schema_dict(feature_list):


    schema_dict = create_schema_dict_from_feature_list(feature_list)

    return schema_dict




def create_features(data,LOAN_TYPE_DICT, 
                    ACCOUNT_DPD_MAPPING_DICT,ACCOUNT_OWNERSHIP_MAPPING_DICT,
                    NUMERICAL_INTERESTING_DICT,
                    CLUSTERING_NUMERICAL_INTERESTING_DICT,
                    SCHEMA_DICT):
    

    error_dict = {}
    dfs_features = pd.DataFrame()
    # STEP 2: Preprocess cibil data for feature creation
    
    try:
        try:
        
            preprocessed_data = preprocess_data(data,
                                                LOAN_TYPE_DICT, 
                                                ACCOUNT_DPD_MAPPING_DICT,ACCOUNT_OWNERSHIP_MAPPING_DICT,
                                                NUMERICAL_INTERESTING_DICT,SCHEMA_DICT)
            

        except Exception as e:
            # raise e
            error_dict.update(
                {constants.error_message_str: str(RawDataPreprocessFailedException(str(e)))})
            return dfs_features,error_dict


        #STEP 2: Creating clustering features data

        try:
            if 'acc_enquiry_clustered' in SCHEMA_DICT or 'agg_enquiry_clustered' in SCHEMA_DICT:
                preprocessed_data['acc_enquiry_clustered'],preprocessed_data['agg_enquiry_clustered'] = create_clustering_features_data(
                    preprocessed_data,CLUSTERING_NUMERICAL_INTERESTING_DICT)

                print('Clustering DFS features data created!')

        except Exception as e:
            # raise e
            error_dict.update(
                {constants.error_message_str: str(ClusteringDFSFeatureCreationFailedException(str(e)))})
            return dfs_features,error_dict
        

        try:
           
            dfs_features = create_dfs_features(preprocessed_data,SCHEMA_DICT)
            
            dfs_features = dfs_features.merge(preprocessed_data['header'][['enquiryControlNumber','CIBIL_Report_Date']],
                                                how='inner',suffixes=('','_y'),
                                                on='enquiryControlNumber')


            print('DFS features created!')
            return dfs_features,error_dict
        
        except Exception as e:
            # raise e
            error_dict.update(
                {constants.error_message_str: str(DFSFeatureCreationFailedException(str(e)))})
            return dfs_features,error_dict
        
    except Exception as e:
        # raise e
        error_dict.update(
                {constants.error_message_str: str(e)})
        return dfs_features,error_dict


    
def create_features_single_json(data,LOAN_TYPE_DICT, 
                    ACCOUNT_DPD_MAPPING_DICT,ACCOUNT_OWNERSHIP_MAPPING_DICT,
                    NUMERICAL_INTERESTING_DICT,
                    CLUSTERING_NUMERICAL_INTERESTING_DICT,
                    SCHEMA_DICT):
    

    error_dict = {}
    dfs_features = pd.DataFrame()
    # STEP 2: Preprocess cibil data for feature creation

    try:
        try:
        
            preprocessed_data = preprocess_data(data,
                                                LOAN_TYPE_DICT, 
                                                ACCOUNT_DPD_MAPPING_DICT,ACCOUNT_OWNERSHIP_MAPPING_DICT,
                                                NUMERICAL_INTERESTING_DICT,SCHEMA_DICT)

        except Exception as e:
            error_dict.update(
                {constants.error_message_str: str(RawDataPreprocessFailedException(str(e)))})
            return dfs_features,error_dict


        #STEP 2: Creating clustering features data

        try:
            if 'acc_enquiry_clustered' in SCHEMA_DICT or 'agg_enquiry_clustered' in SCHEMA_DICT:
                preprocessed_data['acc_enquiry_clustered'],preprocessed_data['agg_enquiry_clustered'] = create_clustering_features_data(
                    preprocessed_data,CLUSTERING_NUMERICAL_INTERESTING_DICT)

                print('Clustering DFS features data created!')

        except Exception as e:
            error_dict.update(
                {constants.error_message_str: str(ClusteringDFSFeatureCreationFailedException(str(e)))})
            return dfs_features,error_dict
        

        try:
            dfs_features = create_dfs_features_single_json(preprocessed_data,SCHEMA_DICT)
            
            dfs_features = dfs_features.merge(preprocessed_data['header'][['enquiryControlNumber','CIBIL_Report_Date']],
                                                how='inner',suffixes=('','_y'),
                                                on='enquiryControlNumber')


            print('Stage one DFS features created!')
            
            return dfs_features,error_dict
        except Exception as e:
            error_dict.update(
                {constants.error_message_str: str(DFSFeatureCreationFailedException(str(e)))})
            return dfs_features,error_dict
        
    except Exception as e:
        error_dict.update(
                {constants.error_message_str: str(e)})
        return dfs_features,error_dict


def process_data_for_feature_creation(
        dataframe_dict,
        LOAN_TYPE_DICT_,
        ACCOUNT_DPD_MAPPING_DICT_,
        ACCOUNT_OWNERSHIP_MAPPING_DICT_,
        NUMERICAL_INTERESTING_DICT_,
        ):


    LOAN_TYPE_DICT = copy.deepcopy(LOAN_TYPE_DICT_)
    ACCOUNT_DPD_MAPPING_DICT = copy.deepcopy(ACCOUNT_DPD_MAPPING_DICT_)
    ACCOUNT_OWNERSHIP_MAPPING_DICT  = copy.deepcopy(ACCOUNT_OWNERSHIP_MAPPING_DICT_)
    NUMERICAL_INTERESTING_DICT = copy.deepcopy(NUMERICAL_INTERESTING_DICT_)

    
    

    if 'header' in dataframe_dict:
        dataframe_dict['header'] = process_header_df(dataframe_dict['header'])

    
    if 'enquiry' in dataframe_dict:
        dataframe_dict['enquiry'] = process_enquiry_df(
            dataframe_dict['enquiry'], dataframe_dict['header'],
            LOAN_TYPE_DICT,None)
        
        if 'enquiry' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['enquiry'] = make_numerical_interesting_vars(dataframe_dict['enquiry'],
                                                                        NUMERICAL_INTERESTING_DICT['enquiry'])
     
        

       
    if 'account' in dataframe_dict:
       

        dataframe_dict['account'] = process_account_df(
            dataframe_dict['account'], dataframe_dict['header'],LOAN_TYPE_DICT,
            ACCOUNT_DPD_MAPPING_DICT,ACCOUNT_OWNERSHIP_MAPPING_DICT)


        if 'account' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['account'] = make_numerical_interesting_vars(dataframe_dict['account'],
                                                                        NUMERICAL_INTERESTING_DICT['account'])
            
        #TODO: Optimize process_tl_mom_df
      
        dataframe_dict['tl_mom'] = process_tl_mom_df(dataframe_dict['account'],dataframe_dict['account'].columns)
      
    

    # if 'score' in dataframe_dict:
    #     dataframe_dict['score'], = process_score_df(
    #         dataframe_dict['score'], dataframe_dict['header'])
    
    
    if 'name' in dataframe_dict:
        dataframe_dict['name'] = process_name_df(dataframe_dict['name'],dataframe_dict['header'])

        if 'name' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['name'] = make_numerical_interesting_vars(dataframe_dict['name'],
                                                                        NUMERICAL_INTERESTING_DICT['name'])

    # if 'identification' in dataframe_dict:
    #     dataframe_dict['identification'], RENAME_DICTS['identification'] = process_id_df(
    #         dataframe_dict['identification'], dataframe_dict['header'], RENAME_DICTS['identification'])
    #     cols_for_interesting_values = []

 
    if 'phone' in dataframe_dict:
        dataframe_dict['phone'] = process_phone_df(
            dataframe_dict['phone'])

        if 'phone' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['phone'] = make_numerical_interesting_vars(dataframe_dict['phone'],
                                                                        NUMERICAL_INTERESTING_DICT['phone'])

    if 'address' in dataframe_dict:
        dataframe_dict['address'] = process_address_df(
            dataframe_dict['address'], dataframe_dict['header'])

        if 'address' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['address'] = make_numerical_interesting_vars(dataframe_dict['address'],
                                                                        NUMERICAL_INTERESTING_DICT['address'])
        

    if 'employment' in dataframe_dict:
        dataframe_dict['employment'] = process_employment_df(
            dataframe_dict['employment'], dataframe_dict['header'])
        
        if 'employment' in NUMERICAL_INTERESTING_DICT:
            dataframe_dict['employment'] = make_numerical_interesting_vars(dataframe_dict['employment'],
                                                                        NUMERICAL_INTERESTING_DICT['employment'])
    
    return dataframe_dict



def preprocess_data_for_feature_creation(data,LOAN_TYPE_DICT, 
                    ACCOUNT_DPD_MAPPING_DICT,ACCOUNT_OWNERSHIP_MAPPING_DICT,
                    NUMERICAL_INTERESTING_DICT,
                    CLUSTERING_NUMERICAL_INTERESTING_DICT):
    

    error_dict = {}
    preprocessed_data = None
    # STEP 2: Preprocess cibil data for feature creation
    
    try:
        try:
        
            preprocessed_data = process_data_for_feature_creation(data,
                                                LOAN_TYPE_DICT, 
                                                ACCOUNT_DPD_MAPPING_DICT,ACCOUNT_OWNERSHIP_MAPPING_DICT,
                                                NUMERICAL_INTERESTING_DICT)
            

        except Exception as e:
            # raise e
            error_dict.update(
                {constants.error_message_str: str(RawDataPreprocessFailedException(str(e)))})
            return preprocessed_data,error_dict

      
        #STEP 2: Creating clustering features data

        try:
        
            preprocessed_data['acc_enquiry_clustered'],preprocessed_data['agg_enquiry_clustered'] = create_clustering_features_data(
                preprocessed_data,CLUSTERING_NUMERICAL_INTERESTING_DICT)

            print('Clustering DFS features data created!')



            return preprocessed_data,error_dict
            
        except Exception as e:
            # raise e
            error_dict.update(
                {constants.error_message_str: str(ClusteringDFSFeatureCreationFailedException(str(e)))})
            return preprocessed_data,error_dict
        

    except Exception as e:
        # raise e
        error_dict.update(
                {constants.error_message_str: str(e)})
        return preprocessed_data,error_dict


   





    






