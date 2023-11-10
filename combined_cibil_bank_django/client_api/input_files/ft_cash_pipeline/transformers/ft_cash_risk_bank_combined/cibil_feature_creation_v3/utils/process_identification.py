import numpy as np

from .helper_functions import create_duration_col
import pandas as pd


def process_id_df(id_df, header_df, df_rename_dict):
    print('processing id data')

    # creating PAN features
    if ('idValue' in id_df.columns) & ('Id_Type' in id_df.columns):
        id_df['pan'] = id_df['idValue'].str[3:4].where(
            id_df['Id_Type'] == '01', np.nan)
        df_rename_dict['pan'] = 'pan'
        # id_df = id_df.drop(columns=['idValue'])
        if 'idValue' in df_rename_dict:
            del df_rename_dict['idValue']

        df_rename_dict['pan'] = 'pan'

    # creating duration lenght columns from datetime columns

    if 'enquiryControlNumber' in id_df.columns:

        id_df = id_df.merge(header_df, left_on='enquiryControlNumber',
                            right_on='enquiryControlNumber')
    else:
        raise ValueError('enquiryControlNumber not present in identification')

    id_df, cols_created = create_duration_col(
        id_df, ['CIBIL_Report_Date'], [
            'Issue_Date', 'Expiry_Date'], df_rename_dict)

    cols_created.append('CIBIL_Report_Date')
    for col in cols_created:
        df_rename_dict[col] = col

    # if Issue_Date is not available for a record , impute with
    # CIBIL_Report_Date
    if 'Issue_Date' in id_df.columns:
        id_df['Issue_Date'] =\
            id_df['Issue_Date'].where(id_df['Issue_Date'].notnull(),
                                      id_df['CIBIL_Report_Date'])
    else:
        id_df['Issue_Date'] = pd.to_datetime('1790-07-17', format='%Y-%m-%d')
        df_rename_dict['Issue_Date'] = 'Issue_Date'

    cols_to_remove = ['Expiry_Date', 'CIBIL_Report_Date']

    # cols_to_remove = list(set(cols_to_remove).intersection(set(id_df.columns)))
    # id_df = id_df.drop(columns=cols_to_remove)

    for col in cols_to_remove:
        if col in df_rename_dict:
            del df_rename_dict[col]

    id_df.index.rename('Identification_Index', inplace=True)
    id_df = id_df.reset_index()

    df_rename_dict['Identification_Index'] = 'Identification_Index'

    return id_df, df_rename_dict
