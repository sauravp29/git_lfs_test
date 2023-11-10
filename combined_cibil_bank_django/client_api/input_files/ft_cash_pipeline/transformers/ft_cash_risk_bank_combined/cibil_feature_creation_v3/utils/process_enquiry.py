

import numpy as np
import pandas as pd
import warnings

def process_enquiry_df(enquiry_df, header_df,loan_type_dict,enquiry_schema_dict):

    print('processing enquiry data')


    try:
        if 'Date_of_Enquiry' in enquiry_df.columns:
            enquiry_df = enquiry_df.loc[enquiry_df['Date_of_Enquiry'].notnull(), :]
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    

    try:
        # adding CIBIL_Report_Date
        enquiry_df = enquiry_df.merge(header_df,on='enquiryControlNumber',
                                    how='inner',suffixes=('', '_y'))
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:
        loan_type_replace_dict = {}

        for key,value in loan_type_dict.items():
            for val in value:
                loan_type_replace_dict[val] = key
        

        enquiry_df['Enquiry_Purpose_Cleaned'] = enquiry_df['Enquiry_Purpose'].map(loan_type_replace_dict)
        enquiry_df['Enquiry_Purpose_Cleaned'] = enquiry_df['Enquiry_Purpose_Cleaned'].fillna('others')
        enquiry_df['Enquiry_Purpose'] = enquiry_df['Enquiry_Purpose'].replace(loan_type_replace_dict)

        
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:
        enquiry_df['Days_Since_Enquiry'] = (enquiry_df['CIBIL_Report_Date'] - enquiry_df['Date_of_Enquiry']).dt.days

    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    
    try:
        enquiry_df['Enquiring_Member_Short_Name'] = enquiry_df['Enquiring_Member_Short_Name'].fillna('NOT DISCLOSED')
        enquiry_df['Enquiring_Member_Short_Name'] = enquiry_df['Enquiring_Member_Short_Name'].replace({'nan':'NOT DISCLOSED'})
        enquiry_df['Enquiring_Member_Short_Name'] = enquiry_df['Enquiring_Member_Short_Name'].str.upper()

    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    
    try:
        
        enquiry_df = enquiry_df.sort_values('Date_of_Enquiry', ascending=True,kind='mergesort')
        enquiry_df = enquiry_df.reset_index(drop=True)

    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    try:
        
        enquiry_groupby_df = enquiry_df.groupby('enquiryControlNumber',sort=False)

        enquiry_df = enquiry_groupby_df.apply(
            lambda x: x.assign(
                Num_of_Days_between_Enquiries=(
                    x.Date_of_Enquiry.subtract(
                        x.Date_of_Enquiry.shift())).dt.days))
        
        
        if enquiry_df.index.nlevels > 1:
            lvl = [i for i in range(1, enquiry_df.index.nlevels)]
            enquiry_df = enquiry_df.reset_index(lvl, drop=True)
        

    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    
    enquiry_df.index.rename('enq_u_id', inplace=True)
    enquiry_df = enquiry_df.reset_index(drop=False)

    return enquiry_df





