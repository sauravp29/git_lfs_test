

import pandas as pd
import numpy as np

from .clustering_utils import (feature_creation_account_df_processing,feature_creation_clustered_df_processing)


from .clustering_logic import (preprocess_account_for_cluster,
                                make_enq_cluster_date_preprocessing,make_enquiry_clusters,agg_cluster_date,
                               find_tradeline_date_alternative,agg_cluster_date_advanced)


def generate_account_and_cluster_df(date_sep,thres_days,amt_thres,dod_buffer,
                                    data_vintage_threshold,
                                    enq_data,acc_data,header_data):
    

   
    #Account Data Preprocessing
    acc_data,enq_data = preprocess_account_for_cluster(enq_data=enq_data,
                                                       account_data=acc_data,
                                                       data_vintage_threshold=data_vintage_threshold)

  
    
    # Enquiry Preprocssing And Making Clusters
    enq_groupby = make_enq_cluster_date_preprocessing(enq_data,
                                                      groupby_col=['enquiryControlNumber'],
                                                      date_sep=date_sep)
 
    enq_clustered_df = enq_groupby.apply(make_enquiry_clusters)
    enq_clustered_df = enq_clustered_df.reset_index(drop=True)
    agg_enq_clustered_df = agg_cluster_date(enq_clustered_df,groupby_col=['enquiryControlNumber','cluster_assigned'])
    

    agg_enq_clustered_df_groupby = agg_enq_clustered_df.groupby('enquiryControlNumber')
    acc_data_groupby = acc_data.groupby('enquiryControlNumber')
    enq_clustered_df_groupby  = enq_clustered_df.groupby('enquiryControlNumber')
    
    acc_test_big = []
    enq_test_big = []
    
    
    unique_acc_groups = set(acc_data_groupby.groups.keys())
    for name,group in agg_enq_clustered_df_groupby:
        enq_clustered_df_subset = enq_clustered_df_groupby.get_group(name)
        if name in unique_acc_groups:
            acc_cibil_subset = acc_data_groupby.get_group(name)
       
            
            
            alt_test,enq_test = find_tradeline_date_alternative(cluster_df_=group,
                                                                account_df_=acc_cibil_subset,
                                                                enq_clustered_df_subset_=enq_clustered_df_subset,
                                                                thres_days=thres_days,amt_thres=amt_thres,
                                                                dod_buffer=dod_buffer,break_cluster=True
                                                                )
            
            acc_test_big.append(alt_test)
            enq_test_big.append(enq_test)
        else:
            enq_test_big.append(enq_clustered_df_subset)

    
    acc_test_big_df = pd.concat(acc_test_big,ignore_index=True,axis=0)
    enq_test_big_df = pd.concat(enq_test_big,ignore_index=True,axis=0)
    
    agg_enq_clustered_df = agg_cluster_date_advanced(enq_test_big_df,groupby_col=['enquiryControlNumber','cluster_assigned'])
    
    acc_test_big_df = acc_test_big_df.merge(agg_enq_clustered_df,on=['enquiryControlNumber','cluster_assigned'],
                                            how='left',suffixes=('','_y'))
    
    acc_test_big_df['abs'] = (acc_test_big_df['Cluster_Mean_Enquiry_Amount'] - acc_test_big_df['High_Credit_Sanctioned_Amount']).abs()
    acc_test_big_df['abs']  = acc_test_big_df['abs']/acc_test_big_df['High_Credit_Sanctioned_Amount']
    not_same_type_not_same_amt = (acc_test_big_df['Account_Type'] != acc_test_big_df['Cluster_Enquiry_Type']) & (acc_test_big_df['abs'] > amt_thres)
    

    acc_test_big_df.loc[not_same_type_not_same_amt,'cluster_assigned'] = np.nan
    
    merge_cols = ['max_DOE', 'min_DOE',
       'Cluster_Size', 'Cluster_Enquiry_Type', 'Cluster_Mean_Enquiry_Amount',
       'Cluster_Max_Enquiry_Amount', 'Cluster_Min_Enquiry_Amount', 'abs']
    acc_test_big_df.loc[acc_test_big_df['cluster_assigned'].isna(),merge_cols] = np.nan

    agg_enq_clustered_df = agg_enq_clustered_df.merge(acc_test_big_df[['enquiryControlNumber','cluster_assigned','tl_u_id']],
                                                      on=['enquiryControlNumber','cluster_assigned'],
                                                      how='left',suffixes=('','_y'))
    
    agg_enq_clustered_df = agg_enq_clustered_df.drop_duplicates(subset=['enquiryControlNumber','cluster_assigned'],keep='first')
    
    agg_enq_clustered_df = agg_enq_clustered_df.merge(header_data,on='enquiryControlNumber',how='left',suffixes=('','_y'))
    
    
    return acc_test_big_df,agg_enq_clustered_df




def generate_clustering_features_data(header_data_,acc_data_,enq_data_,date_sep=15,thres_days=90,
                    amt_thres=0.4,dod_buffer=3,num_interesting_value_dict=None,
                   data_vintage_threshold=40):
    


    header_data = header_data_.copy(deep=True)
    acc_data = acc_data_.copy(deep=True)
    enq_data = enq_data_.copy(deep=True)
    acc_cluster_matched_df,agg_enq_clustered_df = generate_account_and_cluster_df(date_sep=date_sep,
                                                                                    thres_days=thres_days,
                                                                                    amt_thres=amt_thres,
                                                                                    dod_buffer=dod_buffer,
                                                                                    data_vintage_threshold=data_vintage_threshold,
                                                                                    enq_data=enq_data,
                                                                                    acc_data=acc_data,
                                                                                    header_data=header_data)
    
    acc_cluster_matched_df = feature_creation_account_df_processing(acc_cluster_matched_df,
                                                                    num_interesting_value_dict['account'])
    
    agg_enq_clustered_df = feature_creation_clustered_df_processing(agg_enq_clustered_df,num_interesting_value_dict['enquiry'])



    return acc_cluster_matched_df,agg_enq_clustered_df

   
  





            

