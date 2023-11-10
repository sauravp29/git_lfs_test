import pandas as pd
import warnings

def process_header_df(header_df):

    print('processing header data')
    # dropping duplicates based on enquiryControlNumber
    if 'enquiryControlNumber' in header_df.columns:
        header_df = header_df.drop_duplicates(
            subset=['enquiryControlNumber'], keep='first')
    else:
        raise ValueError('enquiryControlNumber not present in header')


    # using todays date if CIBIL_Report_Date is not present
    if 'CIBIL_Report_Date' not in header_df.columns:
        warnings.warn(
        f'CIBIL_Report_Date is Missing in header. Creating new column with todays date')
        header_df['CIBIL_Report_Date'] = pd.to_datetime('today').normalize()
    #TODO: check this once
    # else:
    #     # QUESTION : what happens if CIBIL_Report_Date not in dfd?
    #     raise ValueError('CIBIL_Report_Date not present in header')

    return header_df
