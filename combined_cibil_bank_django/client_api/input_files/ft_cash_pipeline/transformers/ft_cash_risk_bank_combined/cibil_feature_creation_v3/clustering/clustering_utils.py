import pandas as pd
import copy


def make_numerical_interesting_vars(df, col_dict,labels = None, new_col_names=None):

    for col,bins in col_dict.items():
        if labels == None:
            label = [str(bins[i])+' to '+str(bins[i+1]) for i in range(len(bins)-1)]
        else:
            label = labels[col]
        
        if col in df.columns:
            if new_col_names == None:
                new_col_name = col + "_categorical"
            else:
                new_col_name = new_col_names[col]
            
            if bins[0] == 0:
                bins[0] = -1
            
            df[new_col_name] = pd.cut(df[col], bins, labels=label)
          
        else:
            print(f'{col} column not found during making numeric interesting variable')
    
                
    return df



def feature_creation_clustered_df_processing(agg_enquiry_df,make_num_dict):

    agg_enquiry_df['Cluster_TL_Found'] = agg_enquiry_df['tl_u_id'].notna()
    agg_enquiry_df['Cluster_TL_Found'] = agg_enquiry_df['Cluster_TL_Found'].astype(float)
    agg_enquiry_df['Cluster_Intra_Cluster_Distance'] = (agg_enquiry_df['max_DOE'] - agg_enquiry_df['min_DOE']).dt.days

    agg_enquiry_df = make_numerical_interesting_vars(agg_enquiry_df,make_num_dict)

    return agg_enquiry_df




def feature_creation_account_df_processing(account_df,make_num_dict):


    #creating extra columns for feature creations

    account_df['Cluster_Found'] = account_df['cluster_assigned'].notna()
    account_df['Cluster_Found'] = account_df['Cluster_Found'].astype(float)
    account_df['Cluster_Intra_Cluster_Distance'] = (account_df['max_DOE'] - account_df['min_DOE']).dt.days
    account_df['Cluster_Amount_Deviation'] = ((account_df['Cluster_Mean_Enquiry_Amount'] - account_df['High_Credit_Sanctioned_Amount']).abs())/account_df['High_Credit_Sanctioned_Amount']
    account_df['Cluster_Loan_Days_Since_Last_Enquiry'] = (account_df['Date_Opened_Disbursed'] -  account_df['max_DOE']).dt.days
    account_df['Cluster_Loan_Days_Since_Last_Enquiry'] = account_df['Cluster_Loan_Days_Since_Last_Enquiry'].where(account_df['Cluster_Loan_Days_Since_Last_Enquiry'] > 0,0)
    account_df['Cluster_Loan_Days_Since_First_Enquiry'] = (account_df['Date_Opened_Disbursed'] -  account_df['min_DOE']).dt.days
    account_df['Cluster_Loan_Days_Since_First_Enquiry'] = account_df['Cluster_Loan_Days_Since_First_Enquiry'].where(account_df['Cluster_Loan_Days_Since_First_Enquiry']> 0,0)
    account_df['Cluster_Type_Matching'] = account_df['Cluster_Enquiry_Type'] == account_df['Account_Type']
    account_df['Cluster_Type_Matching'] = account_df['Cluster_Type_Matching'].astype(float)

    account_df = make_numerical_interesting_vars(account_df,make_num_dict)

    
    return account_df

