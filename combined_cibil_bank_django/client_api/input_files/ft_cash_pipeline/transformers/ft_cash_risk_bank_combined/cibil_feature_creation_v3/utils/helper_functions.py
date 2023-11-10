import numpy as np
import pandas as pd
import copy
import math
import pickle

def load_dict_helper(dict_object):
        
    if isinstance(dict_object, dict):
        return copy.deepcopy(dict_object)

    elif isinstance(dict_object, str):
        with open(dict_object,'rb') as f:
            return pickle.load(f)
    
    elif dict_object is None:
        return dict_object
               
      
            

def create_ratios(df, ratio_tuple_list):

    for ratio_col_1,ratio_col_2 in ratio_tuple_list:
        if ratio_col_1 in df.columns and ratio_col_2 in df.columns:
            name = 'frac_'+ratio_col_1+f'__{ratio_col_2}'
            df[name] = df[ratio_col_1]/df[ratio_col_2]
            df[name] = df[name].replace([-np.inf,np.inf],np.nan)
        else:
            print(f'{ratio_col_1} or {ratio_col_2} is not present in df columns')
        
    return df


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


def make_categorical_interesting_vars(df, col_dict,
                                      default="unknown"):

    created_cols = []
 
    for col, cat_map in col_dict.items():
        if col in df.columns:
            conditions = [df[col].isin(v) for k, v in cat_map.items()]
            choices = cat_map.keys()
            new_col_name = col + "_bucket"
            df[new_col_name] = np.select(conditions, choices, default=default)
            created_cols.append(new_col_name)

    return df


def divide_to_chunks(dfd, size=5000,n_jobs=1):

    dfd_chunk = []
    total = dfd["header"].shape[0]
    num_split = math.ceil(total / size)

    if n_jobs > num_split:
        num_split = n_jobs

    header_split = np.array_split(dfd["header"], num_split)

    print(f'data chunked into {num_split} parts')
  
    for split in header_split:
        dfd_t = {}
        dfd_t["header"] = split
        dfd_t["header"] = dfd_t["header"].reset_index(drop=True)

        for k in dfd:
            if k != 'header':
                mask = dfd[k]["enquiryControlNumber"].isin(
                    dfd_t["header"].enquiryControlNumber)
                dfd_t[k] = dfd[k][mask]
                dfd_t[k] = dfd_t[k].reset_index(drop=True)
            # print(dfd_t[k].shape)

        dfd_chunk.append(dfd_t)
        # print(dfd_t["header"].shape)

    return dfd_chunk



def create_schema_dict_from_feature(feat):
        
        feature_schema_dict = {}
        feature_schema_dict['feature_name'] = feat

        if  '(' in feat:
            primary_agg = feat.split('(')[0]
            feature_schema_dict['primary_agg'] = primary_agg
        else:
            raise ValueError(f'Primary aggregate not found in feature {feat}')
        

        if '_months' in feat:

            month = int(feat.split('__')[-1].split('_')[0])
            month_split = '__'+str(month)+'_months'
            feat_name = feat.split(month_split)[0]
        else:
            month = None
            month_split = ''
            feat_name = feat
        

        feature_schema_dict['time_subset'] = month
   

        if 'WHERE'in feat_name:
            where_part = feat_name.split(" WHERE ")[-1]
            where_col = where_part.split(' = ')[0]
            where_col = where_col.strip('(),_')
            where_val = where_part.split(' = ')[-1][:-1]
            where_val = where_val.strip('(),_')

            feat_name = feat_name.split(" WHERE ")[0]


            if where_col in ['Cluster_Found','Cluster_Type_Matching','Cluster_TL_Found']:
                where_val = int(where_val)
    
        else:
            where_col = None
            where_val = None
        

        feature_schema_dict['where_column'] = where_col
        feature_schema_dict['where_value'] = where_val


        if '.' in feat_name:
            agg_column = feat_name.split('.')[-1]
            agg_column = agg_column.strip('(),_')
            feature_schema_dict['agg_column'] = agg_column

        else:
            raise ValueError(f'aggregate column not found in feature {feat}')



        return feature_schema_dict



def dataframe_to_schema_dict(schema_features_dict_df):

    feature_schema_dict = {}
    columns_used = set()
    if len(schema_features_dict_df) == 0:
        return feature_schema_dict,columns_used
    
    schema_features_dict_df = pd.DataFrame(schema_features_dict_df)
    schema_features_dict_df = schema_features_dict_df.replace(np.nan,None)

    total = 0
    
    if schema_features_dict_df.shape[0] > 0:
        columns_used = set(schema_features_dict_df[schema_features_dict_df['agg_column'].notna()]['agg_column'].to_list()).union(
                    set(schema_features_dict_df[schema_features_dict_df['where_column'].notna()]['where_column'].to_list()))
        columns_used.update({'CIBIL_Report_Date','enquiryControlNumber'})
        for new_name,prim_agg,agg_col,where_value,where_col,time in zip(schema_features_dict_df['feature_name'],
                                                                schema_features_dict_df['primary_agg'],
                                                                schema_features_dict_df['agg_column'],
                                                                schema_features_dict_df['where_value'],
                                                                schema_features_dict_df['where_column'],
                                                                    schema_features_dict_df['time_subset']):
            
            
            if time not in feature_schema_dict:
                if time is not None:
                    time = int(time)
                feature_schema_dict[time] = {}
                
            
            if (where_col,where_value) not in feature_schema_dict[time]:
                feature_schema_dict[time][(where_col,where_value)] = {}
                
        
                

            feature_schema_dict[time][(where_col,where_value)][new_name] = (agg_col,prim_agg.lower())
            total +=1


    print(f'schema dict created for {total} features ')

    return feature_schema_dict,columns_used



def create_schema_dict(feature_list):
    
    
    schema_features_dict_df = []
    ratio_dict = {}

    for feat in feature_list:

        if 'frac_' in feat[:5]:
            feat_split_1 = feat.split('___')[0][5:]
            feat_split_2 = feat.split('___')[1]
            schema_features_dict_df.append(create_schema_dict_from_feature(feat_split_1))
            schema_features_dict_df.append(create_schema_dict_from_feature(feat_split_2))
            ratio_dict[feat] = (feat_split_1,feat_split_2)
        else:
            schema_features_dict_df.append(create_schema_dict_from_feature(feat))


    feature_schema_dict,columns_used = dataframe_to_schema_dict(schema_features_dict_df)
       
    return [feature_schema_dict,ratio_dict,columns_used]



def create_stage_2_schema_dict_from_feature(feat):
        
        
        feature_schema_dict = {}
        feature_schema_dict['feature_name'] = feat

        if '(' in feat:
            primary_agg = feat.split('(')[0]
            feature_schema_dict['primary_agg'] = primary_agg
            feat_name = feat[len(primary_agg):]
        else:
            raise ValueError(f'Primary aggregate not found in feature {feat}')
        

        if ')' in feat:
            tl_mom_feature = feat_name.split(')')[0]
            tl_mom_feature = tl_mom_feature +')'
            tl_mom_feature = tl_mom_feature[12:]
            tl_mom_feature_index = feat_name.index(tl_mom_feature) + len(tl_mom_feature)
            feat_name = feat_name[tl_mom_feature_index:]

        else:
            raise ValueError(f'tl mom feature not found in feature {feat}')
        

        if '_months' in feat:

            month = int(feat_name.split('__')[-1].split('_')[0])
            month_split = '__'+str(month)+'_months'
            feat_name = feat_name.split(month_split)[0]
            tl_mom_feature+= month_split
        else:
            month = None
            month_split = ''
        
        

        feature_schema_dict['agg_column'] = tl_mom_feature 
        tl_mom_feature_schema_dict = create_schema_dict_from_feature(tl_mom_feature)
         
        
        feature_schema_dict['time_subset'] = month
   

        if 'WHERE'in feat_name:
            where_part = feat_name.split(" WHERE ")[-1]
            where_col = where_part.split(' = ')[0]
            where_col = where_col.strip('(),_')
            where_val = where_part.split(' = ')[-1][:-1]
            where_val = where_val.strip('(),_')

            feat_name = feat_name.split(" WHERE ")[0]


            if where_col in ['Cluster_Found','Cluster_Type_Matching','Cluster_TL_Found']:
                where_val = int(where_val)
    
        else:
            where_col = None
            where_val = None
        

        feature_schema_dict['where_column'] = where_col
        feature_schema_dict['where_value'] = where_val


        return feature_schema_dict,tl_mom_feature_schema_dict



def create_stage_2_schema_dict(dfs_features):

    schema_features_dict_df = []
    tl_mom_features_dict_df = []
    ratio_dict = {}

    for feat in dfs_features:
        if 'frac_' in feat[:5]:
            feat_split_1 = feat.split('___')[0][5:]
            feat_split_2 = feat.split('___')[1]
            account_schema_dict_1,tl_mom_schema_dict_1 = create_stage_2_schema_dict_from_feature(feat_split_1)
            schema_features_dict_df.append(account_schema_dict_1)
            tl_mom_features_dict_df.append(tl_mom_schema_dict_1)
            account_schema_dict_2,tl_mom_schema_dict_2 = create_stage_2_schema_dict_from_feature(feat_split_2)
            schema_features_dict_df.append(account_schema_dict_2)
            tl_mom_features_dict_df.append(tl_mom_schema_dict_2)
            ratio_dict[feat] = (feat_split_1,feat_split_2)
        else:
            account_schema_dict_1,tl_mom_schema_dict_1 = create_stage_2_schema_dict_from_feature(feat)
            schema_features_dict_df.append(account_schema_dict_1)
            tl_mom_features_dict_df.append(tl_mom_schema_dict_1)
    
    
    tl_features_schema_dict,tl_columns_used = dataframe_to_schema_dict(schema_features_dict_df)
    tl_mom_features_schema_dict,tl_mom_columns_used = dataframe_to_schema_dict(tl_mom_features_dict_df)

    return tl_features_schema_dict,tl_mom_features_schema_dict,ratio_dict,tl_columns_used,tl_mom_columns_used





def create_schema_dict_from_feature_list(feature_list_):
    

    feature_list = copy.deepcopy(feature_list_)
    schema_dict = {}


    stage_2_dfs_features = [feat for feat in feature_list if (('tl_mom' in feat) and ('account_df' in feat))] 
    schema_dict['stage_2_dfs_features'] = create_stage_2_schema_dict(stage_2_dfs_features)

    for feat in stage_2_dfs_features:
        feature_list.remove(feat)


    tl_mom_df_features = [feat for feat in feature_list if 'tl_mom' in feat]
    schema_dict['tl_mom'] = create_schema_dict(tl_mom_df_features)


    account_df_features = [feat for feat in feature_list if 'account_df' in feat]
    schema_dict['account'] = create_schema_dict(account_df_features)
    schema_dict['account'][2] = schema_dict['account'][2].union({'Account_Type',
                                                                'Date_Opened_Disbursed',
                                                                'High_Credit_Sanctioned_Amount',
                                                                'tl_u_id'})

            
    enquiry_df_features = [feat for feat in feature_list if 'enquiry_df' in feat]
    schema_dict['enquiry'] = create_schema_dict(enquiry_df_features)
    schema_dict['enquiry'][2] = schema_dict['enquiry'][2].union({'Enquiry_Purpose',
                                                                'Date_of_Enquiry','Enquiry_Amount',
                                                                    'enq_u_id'})


    address_df_features = [feat for feat in feature_list if 'address_df' in feat]
    schema_dict['address'] = create_schema_dict(address_df_features)


    phone_df_features = [feat for feat in feature_list if 'phone_df' in feat]
    schema_dict['phone'] = create_schema_dict(phone_df_features)

    employment_df_features = [feat for feat in feature_list if 'employment_df' in feat]
    schema_dict['employment'] = create_schema_dict(employment_df_features)

    name_df_features = [feat for feat in feature_list if 'name_df' in feat]
    schema_dict['name'] = create_schema_dict(name_df_features)

    agg_enquiry_clustered_df_features = [feat for feat in feature_list if 'agg_enquiry_clustered_df' in feat]
    schema_dict['agg_enquiry_clustered'] = create_schema_dict(agg_enquiry_clustered_df_features)
  

    acc_enquiry_clustered_df_features = [feat for feat in feature_list if 'acc_enquiry_clustered_df' in feat]
    schema_dict['acc_enquiry_clustered'] = create_schema_dict(acc_enquiry_clustered_df_features)
   

    final_schema_dict = {}
    for dataset in schema_dict:
        if schema_dict[dataset][0]:
            final_schema_dict[dataset] = schema_dict[dataset]
    
   
    if 'stage_2_dfs_features' in final_schema_dict:
        if 'account' not in final_schema_dict:
            final_schema_dict['account'] = [{},{},set()]
        final_schema_dict['account'][2] = final_schema_dict['account'][2].union(final_schema_dict['stage_2_dfs_features'][3])
        final_schema_dict['account'][2] = final_schema_dict['account'][2].union(final_schema_dict['stage_2_dfs_features'][4])
        final_schema_dict['account'][2] = final_schema_dict['account'][2].union({'Account_Type',
                                                                'Date_Opened_Disbursed',
                                                                'High_Credit_Sanctioned_Amount',
                                                                'tl_u_id'})

        if 'tl_mom' not in final_schema_dict:
            final_schema_dict['tl_mom'] = [{},{},set()]
        final_schema_dict['tl_mom'][2] = final_schema_dict['tl_mom'][2].union(final_schema_dict['stage_2_dfs_features'][4])
        
    
    
    if 'acc_enquiry_clustered' in final_schema_dict:
        if 'account' not in final_schema_dict:
            final_schema_dict['account'] = [{},{},set()]
        final_schema_dict['account'][2] = final_schema_dict['account'][2].union(final_schema_dict['acc_enquiry_clustered'][2])
        final_schema_dict['account'][2] = final_schema_dict['account'][2].union({'Account_Type',
                                                                'Date_Opened_Disbursed',
                                                                'High_Credit_Sanctioned_Amount',
                                                                'tl_u_id'})
    
    if 'tl_mom' in final_schema_dict:
        if 'account' not in final_schema_dict:
            final_schema_dict['account'] = [{},{},set()]
        final_schema_dict['account'][2] = final_schema_dict['account'][2].union(final_schema_dict['tl_mom'][2])
        final_schema_dict['account'][2] = final_schema_dict['account'][2].union({'Account_Type',
                                                                'Date_Opened_Disbursed',
                                                                'High_Credit_Sanctioned_Amount',
                                                                'tl_u_id'})

    

    if 'agg_enquiry_clustered' in final_schema_dict:
        if 'enquiry' not in final_schema_dict:
            final_schema_dict['enquiry'] = [{},{},set()]
        
        final_schema_dict['enquiry'][2] = final_schema_dict['enquiry'][2].union(final_schema_dict['agg_enquiry_clustered'][2])
        final_schema_dict['enquiry'][2] = final_schema_dict['enquiry'][2].union({'Enquiry_Purpose',
                                                                                    'Date_of_Enquiry','Enquiry_Amount',
                                                                                'enq_u_id'})

    return final_schema_dict













    
