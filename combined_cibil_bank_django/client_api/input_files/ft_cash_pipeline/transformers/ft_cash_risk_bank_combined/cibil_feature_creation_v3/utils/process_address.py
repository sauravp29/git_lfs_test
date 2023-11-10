import numpy as np
import pandas as pd
import warnings


def process_address_df(address_df, header_df):
    print('processing address data')

    # creating Pin_Code based features
    try:

        if 'Pin_Code' in address_df.columns:
            address_df['Pin_Code'] = address_df['Pin_Code'].astype(str).replace(
                {'NA': '', 'NaN': '', 'Na': '', 'NA': '', 'na': '', 'nan': ''})
            address_df['Pin_Code'] = address_df['Pin_Code'].astype(
                str).replace({i: '' for i in (', ')}, regex=True)
            
            address_df['Pin_Code'] = address_df['Pin_Code'].str[3:]
            address_df['Pin_Code'] = pd.to_numeric(
                address_df['Pin_Code'], errors='coerce')

    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    try:
        address_df = address_df.merge(header_df, left_on='enquiryControlNumber',
                                    right_on='enquiryControlNumber',how='inner',suffixes=('','_y'))
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    

    try:
        address_df['Duration_Living_Address'] = (address_df['CIBIL_Report_Date'] - address_df['Date_Reported']).dt.days
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    

    try:
        address_df = address_df.sort_values(by='Date_Reported', ascending=False,kind='mergesort')
        address_df = address_df.reset_index(drop=True)

    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    return address_df
