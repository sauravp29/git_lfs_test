import pandas as pd
import numpy as np
import copy




def preprocess_account_for_cluster(enq_data,account_data,data_vintage_threshold):


    # merging consumer and personal loan accountt type and enquiries purpose
    account_data['Account_Type'] = account_data['Account_Type'].replace({'consumer':'personal'})
    enq_data['Enquiry_Purpose'] = enq_data['Enquiry_Purpose'].replace({'consumer':'personal'})

    # REMOVING GOLD TYPE LOANS and loan against bank deposit
    account_data = account_data[~account_data['Account_Type'].isin(['gold','loan_against_bank_deposits'])]
    enq_data = enq_data[~enq_data['Enquiry_Purpose'].isin(['gold','loan_against_bank_deposits'])]


    enq_data['data_vintage_threshold_date'] = enq_data['CIBIL_Report_Date'] - pd.DateOffset(months=data_vintage_threshold)
    enq_data = enq_data[enq_data['Date_of_Enquiry'] >= enq_data['data_vintage_threshold_date']]


    account_data['data_vintage_threshold_date'] = account_data['CIBIL_Report_Date'] - pd.DateOffset(months=data_vintage_threshold)
    account_data = account_data[account_data['Date_Opened_Disbursed'] >= account_data['data_vintage_threshold_date']]

    min_date_enq = enq_data.groupby('enquiryControlNumber')['Date_of_Enquiry'].min()
    min_date_enq = min_date_enq.reset_index(drop=False)
    min_date_enq['enquiryControlNumber'] = min_date_enq['enquiryControlNumber'].astype(str)

    
    account_data = account_data.merge(min_date_enq,on='enquiryControlNumber',how='left',suffixes=('','_y'))
    account_data = account_data[account_data['Date_Opened_Disbursed'] >= account_data['Date_of_Enquiry']]
    account_data = account_data.sort_values(by=['Date_Opened_Disbursed','High_Credit_Sanctioned_Amount'],
                                            ascending=[True,False],kind='mergesort')
    account_data = account_data.reset_index(drop=True)
    return account_data,enq_data


def make_enq_cluster_date_preprocessing(df,
                                    date_sep=15,
                                    groupby_col = None):

 

    df = df.sort_values(by='Date_of_Enquiry',ascending=True,kind='mergesort')
    df['Date_max'] = df['Date_of_Enquiry'] + pd.DateOffset(days=date_sep)
    df['Date_min'] = df['Date_of_Enquiry'] - pd.DateOffset(days=date_sep)
    df = df.reset_index(drop=True)
    if groupby_col is None:
        return df
    else:
        return df.groupby(groupby_col)



def make_enquiry_clusters(df,date_col='Date_of_Enquiry'):
  
    index_list = df.index.to_list()
    cluster_index = 0
    index_in_cluster = set()
    for index in index_list:
        if index not in index_in_cluster:
            df_subset =  df[(df.loc[index,'Date_max'] >= df[date_col]) & (df[date_col] >= df.loc[index,'Date_of_Enquiry'])]
            df_subset = df_subset[df_subset['Enquiry_Purpose'] == df.loc[index,'Enquiry_Purpose']]
            df_subset = df_subset[~df_subset.index.isin(index_in_cluster)]
            df.loc[df_subset.index,'cluster_assigned'] = cluster_index
            cluster_index += 1
            index_in_cluster.update(df_subset.index.to_list())
    
    return df



def agg_cluster_date(clustered_df,groupby_col='cluster_assigned'):

    clustered_groupby = clustered_df.groupby(groupby_col)

    cluster_agg = clustered_groupby.agg(max_DOE=("Date_of_Enquiry", "max"),
                        min_DOE=('Date_of_Enquiry','min'),
                        Cluster_Size = ('Date_of_Enquiry','count'),
                        Cluster_Enquiry_Type = ('Enquiry_Purpose','first'),
                        Cluster_Mean_Enquiry_Amount = ('Enquiry_Amount','mean'))
                        
    
    
    cluster_agg = cluster_agg.reset_index(drop=False)
    
    return cluster_agg


def agg_cluster_date_advanced(clustered_df,groupby_col='cluster_assigned'):

    clustered_groupby = clustered_df.groupby(groupby_col)

    cluster_agg = clustered_groupby.agg(max_DOE=("Date_of_Enquiry", "max"),
                        min_DOE=('Date_of_Enquiry','min'),
                        Cluster_Size = ('Date_of_Enquiry','count'),
                        Cluster_Enquiry_Type = ('Enquiry_Purpose','first'),
                        Cluster_Mean_Enquiry_Amount = ('Enquiry_Amount','mean'),
                        Cluster_Max_Enquiry_Amount = ('Enquiry_Amount','max'),
                        Cluster_Min_Enquiry_Amount = ('Enquiry_Amount','min'))
    
    
    cluster_agg = cluster_agg.reset_index(drop=False)
    
    return cluster_agg



def break_cluster_into_two(cluster_df_subset,enq_clustered_df_subset,account_df,account_dict,amt_thres):

  
    cluster_df_subset_dict = cluster_df_subset.iloc[0].to_dict()

    account_df_subset = account_df[account_df['cluster_assigned'] == cluster_df_subset_dict['cluster_assigned']]

    account_mapped = True
    if account_df_subset.shape[0] >= 1:
        account_df_subset = account_df_subset.iloc[0].to_dict()
    
    else:
        account_mapped = False
            
    
    clustered_break_subset = enq_clustered_df_subset[enq_clustered_df_subset['cluster_assigned'] == cluster_df_subset_dict['cluster_assigned']]


    if account_mapped:
        day_diff = (account_df_subset['Date_Opened_Disbursed'] - account_dict['Date_Opened_Disbursed']).days
   
        if abs(day_diff) <=3:
            return 'SAME'


    if (cluster_df_subset_dict['max_DOE'] > account_dict['Date_Opened_Disbursed']) and (pd.notnull(account_dict['Date_Opened_Disbursed'])):

        cluster_1 = clustered_break_subset[clustered_break_subset['Date_of_Enquiry'] <= account_dict['Date_Opened_Disbursed']]
      
        if cluster_1.shape[0] < clustered_break_subset.shape[0]:
            return cluster_1['enq_u_id'].to_list()
        
    
    if not pd.isnull(account_dict['High_Credit_Sanctioned_Amount']):
        lower = account_dict['High_Credit_Sanctioned_Amount']*(1 - amt_thres)
        upper = account_dict['High_Credit_Sanctioned_Amount']* (1 + amt_thres)

        cluster_1 = clustered_break_subset[(clustered_break_subset['Enquiry_Amount'] >= lower) & (clustered_break_subset['Enquiry_Amount'] <= upper)]

        if (cluster_1.shape[0]) == 0 :
            return 'NO MAPPING'
        
        if (cluster_1.shape[0] == clustered_break_subset.shape[0]):
            return 'SAME'

        if cluster_1.shape[0] < clustered_break_subset.shape[0]:
            return cluster_1['enq_u_id'].to_list()
        
    return 'NO MAPPING'
        
        
  

def find_best_cluster_to_break(cluster_df_subset,account_dict):
    if not cluster_df_subset.shape[0] == 1:
        
        closest_index = (cluster_df_subset['Cluster_Mean_Enquiry_Amount'] - account_dict['High_Credit_Sanctioned_Amount']).abs().idxmin()

        
        if not pd.isnull(closest_index):
            cluster_df_subset = cluster_df_subset[cluster_df_subset.index == closest_index]
                        
        else:
            cluster_df_subset = cluster_df_subset[cluster_df_subset.index == cluster_df_subset.index[0]]
    
    return cluster_df_subset
    


def find_tradeline_date_alternative(cluster_df_,account_df_,enq_clustered_df_subset_,thres_days=60,amt_thres=0.3,
                                    dod_buffer=3,break_cluster=False):
    

    cluster_df_['max_DOE_alt'] = cluster_df_['max_DOE'] - pd.DateOffset(days=dod_buffer)
    cluster_count = cluster_df_['cluster_assigned'].max()
    cluster_df = cluster_df_.copy(deep=True)
    account_df = account_df_.copy(deep=True)
    enq_clustered_df_subset = enq_clustered_df_subset_.copy(deep=True)  
    account_df['cluster_assigned'] = np.nan
    account_df['min_DOD'] = account_df['Date_Opened_Disbursed'] - pd.DateOffset(days=thres_days)
    
    for account in account_df.index.to_list():
        
        cluster_matched = True

        account_dict = account_df.loc[account].to_dict()

        cluster_df_subset = cluster_df[cluster_df['max_DOE_alt']<= account_dict['Date_Opened_Disbursed']].copy()
        cluster_df_subset = cluster_df_subset[cluster_df_subset['max_DOE_alt']>= account_dict['min_DOD']]

      
        if cluster_df_subset.shape[0] >= 1:

            same_type = False
            # Enquiry Type/ Account Type
            cluster_df_subset_enq_type = cluster_df_subset[cluster_df_subset['Cluster_Enquiry_Type'] == account_dict['Account_Type']]
            
            if cluster_df_subset_enq_type.shape[0] > 1:

                    closest_index = (cluster_df_subset_enq_type['Cluster_Mean_Enquiry_Amount'] - account_dict['High_Credit_Sanctioned_Amount']).abs().idxmin()

                    same_type = True

                    if not pd.isnull(closest_index):
                        final_cluster_df_subset = cluster_df_subset_enq_type[cluster_df_subset_enq_type.index == closest_index]
                    
                    else:
                        final_cluster_df_subset = cluster_df_subset_enq_type[cluster_df_subset_enq_type.index == cluster_df_subset_enq_type.index[0]]
            
            elif cluster_df_subset_enq_type.shape[0] == 0:

                final_cluster_df_subset = cluster_df_subset
        
            else:
                final_cluster_df_subset = cluster_df_subset_enq_type
                same_type = True

            
            
            if final_cluster_df_subset.shape[0] >= 1 and same_type==False:
                
                if not np.isnan(account_dict['High_Credit_Sanctioned_Amount']):
                    final_cluster_df_subset['abs_diff']  = (final_cluster_df_subset['Cluster_Mean_Enquiry_Amount'] - account_dict['High_Credit_Sanctioned_Amount']).abs()
                    final_cluster_df_subset['abs_diff']   = (final_cluster_df_subset['abs_diff'] / account_dict['High_Credit_Sanctioned_Amount']) 
                    final_cluster_df_subset  = final_cluster_df_subset[final_cluster_df_subset['abs_diff'] <= amt_thres] 

                    if final_cluster_df_subset.shape[0] > 1:

                        closest_index = final_cluster_df_subset['abs_diff'].idxmin()

                        if not pd.isnull(closest_index):
                            final_cluster_df_subset = final_cluster_df_subset[final_cluster_df_subset.index == closest_index]
                        else:
                            print('cluster index is null')
                            cluster_matched = False

                    elif final_cluster_df_subset.shape[0] == 0 :

                        cluster_matched = False
                    else:
                        pass

                else:
                    cluster_matched = False
        
        else:
            cluster_matched = False

          
        
        if cluster_matched == False:

            if break_cluster:
               
                
                cluster_df_subset = cluster_df_[cluster_df_['min_DOE']<= account_dict['Date_Opened_Disbursed']].copy(deep=True)
                cluster_df_subset = cluster_df_subset[cluster_df_subset['min_DOE']>= account_dict['min_DOD']]
                cluster_df_subset = cluster_df_subset[cluster_df_subset['Cluster_Enquiry_Type'] == account_dict['Account_Type']]

                if cluster_df_subset.shape[0] >= 1:
                    
                    cluster_count +=1
                    cluster_df_subset = find_best_cluster_to_break(cluster_df_subset,account_dict)

                  

                    enq_id_to_update = break_cluster_into_two(cluster_df_subset,
                                                              enq_clustered_df_subset,
                                                              account_df,account_dict,
                                                              amt_thres)

                    

                    if enq_id_to_update == 'SAME':
                        final_cluster_df_subset = cluster_df_subset
                        cluster_matched = True
                    elif enq_id_to_update == 'NO MAPPING':
                        pass
                    else:
                        # enq_clustered_df_subset['cluster_assigned'].loc[enq_clustered_df_subset['enq_u_id'].isin(enq_id_to_update)] = cluster_count

                        enq_clustered_df_subset['cluster_assigned'] = np.where(enq_clustered_df_subset['enq_u_id'].isin(enq_id_to_update),
                                                                               cluster_count,enq_clustered_df_subset['cluster_assigned'])
                        final_cluster_df_subset = pd.DataFrame(cluster_count,columns=['cluster_assigned'],index=[0])
                        cluster_df_ = agg_cluster_date(enq_clustered_df_subset)
                        cluster_df_['max_DOE_alt'] = cluster_df_['max_DOE'] - pd.DateOffset(days=dod_buffer)
                        cluster_matched = True
                        # cluster_df = cluster_df_[cluster_df_['cluster_assigned'].isin(cluster_df['cluster_assigned'])].copy(deep=True)
                        
            
        if cluster_matched:
            #TODO: only use cluster_id insteaf of final_cluster_df_subset dataframe
            account_df.loc[account,'cluster_assigned'] = final_cluster_df_subset['cluster_assigned'].iloc[0]
           
            if final_cluster_df_subset['cluster_assigned'].iloc[0] in set(cluster_df['cluster_assigned']):
                    cluster_df = cluster_df[cluster_df['cluster_assigned'] != final_cluster_df_subset['cluster_assigned'].iloc[0]]
          
    
    return account_df,enq_clustered_df_subset


        
        

    
      










