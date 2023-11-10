import numpy as np
import pandas as pd
import warnings




def reconcile_datasets_rd(dataframe_dict, RENAME_DICT):
    '''Brings RENAME_DICTS and datasets to same keys level'''

    missing_datasets = [i for i in RENAME_DICT if i not in dataframe_dict]

    missing_mappings_rd = [i for i in dataframe_dict if i not in RENAME_DICT]

 
    if len(missing_datasets):
        raise ValueError(
            f'Mapping provided but corresponding Dataset is missing. Key: {missing_datasets}')

    if len(missing_mappings_rd):
        # If this error is raised then something else is wrong. Since dtypes and rename_dicts are reconciled before
        # datasets and rd, this condition should never evaluate to True. So everything should get caught in the
        # missing_mappings_dt  only
        raise ValueError(
            f'Dataset provided but corresponding Mapping is missing in RENAME_AND_DTYE_DICT. Key: {missing_mappings_rd}')

    for dataset in RENAME_DICT:
        missing_columns = [
            t for t in RENAME_DICT[dataset].values() if t not in dataframe_dict[dataset].columns]

        if len(missing_columns):
            warnings.warn(
        f'Mapping provided but Columns is missing from Dataset {dataset}. Key: {missing_columns}. Creating said columns with null values')
            for col in missing_columns:
                dataframe_dict[dataset][col] = np.nan


    return dataframe_dict


def format_and_rename_cols(df, df_dtypes, df_rename_dict, df_date_dict):
    
    rev_rename_dict = {i: j for j, i in df_rename_dict.items()}
    df = df.rename(columns=rev_rename_dict)
    

    for col in df_rename_dict:
        if col not in df.columns:
            df[col] = np.nan
    
    for col,dtype in df_dtypes.items():

        try:
            if dtype == 'string':
                df[col] = df[col].astype(str)
            
            elif dtype == 'numeric':
                df[col] = pd.to_numeric(df[col],errors='coerce')
            
            elif dtype == 'datetime':
                 
                df[col] = pd.to_datetime(df[col],format = df_date_dict.get(col,df_date_dict['default']),errors='coerce')
                df[col] = df[col].dt.date
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d",errors='coerce')

            else:
                raise ValueError(f'Dtype {dtype} not understood')
            
        except Exception as e:
            print(str(e),col)

    return df

