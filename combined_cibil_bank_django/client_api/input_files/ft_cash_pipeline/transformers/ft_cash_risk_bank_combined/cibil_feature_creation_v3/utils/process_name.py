import numpy as np
import warnings


def process_name_df(name_df,header_df):
    print('processing name data')

    # dropping duplicates based on enquiryControlNumber
    if 'enquiryControlNumber' in name_df.columns:
        name_df = name_df.drop_duplicates(
            subset=['enquiryControlNumber'], keep='first')
    else:
        raise ValueError('enquiryControlNumber not present in name')


    try:
        name_df = name_df.merge(header_df,on='enquiryControlNumber',
                                    how='inner',suffixes=('', '_y'))
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
        
    try:
        name_df['Age']  = name_df['CIBIL_Report_Date'].dt.year - name_df['DOB'].dt.year
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
                
    return name_df
