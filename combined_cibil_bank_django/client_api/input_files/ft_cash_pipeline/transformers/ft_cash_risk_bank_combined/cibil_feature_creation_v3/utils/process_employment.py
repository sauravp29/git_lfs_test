import warnings

def process_employment_df(employment_df,header_df):
    print('processing employment data')

    # dropping duplicates

    try:

        employment_df = employment_df.sort_values(by='Date_Reported', ascending=False,kind='mergesort')
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    
    if 'enquiryControlNumber' in employment_df.columns:
        employment_df = employment_df.drop_duplicates(
            subset=['enquiryControlNumber'], keep='first')

    else:
        raise ValueError('enquiryControlNumber not in employment')
    

    try:

        employment_df = employment_df.merge(header_df, left_on='enquiryControlNumber',
                                    right_on='enquiryControlNumber',how='inner',suffixes=('','_y'))
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    
    try:

        if 'Employment_Type' in employment_df.columns:
            employment_df['Employment_Type'] = employment_df['Employment_Type'].str.upper() 
            employment_df['Self_Employed_Indicator'] = employment_df['Employment_Type'].isin(['SELF EMPLOYED PROFESSIONAL',
                                                                                            'SELF EMPLOYED',
                                                                                            '02','2','03','3'])
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))   
        

    try:
        if 'Monthly_Annual_Indicator' in employment_df.columns:
            employment_df['Monthly_Annual_Indicator'] = employment_df['Monthly_Annual_Indicator'].str.upper()
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    
    try:
        if ('Income' in employment_df.columns) and ('Monthly_Annual_Indicator' in employment_df.columns):

            employment_df['Income'] = employment_df['Income'].where(
                employment_df['Monthly_Annual_Indicator'].isin(['A','NAN']), employment_df['Income'] * 12)
            
             # employment_df['Income'] = employment_df['Income'].where(
        #       employment_df['Monthly_Annual_Indicator'] == 'A', employment_df['Income'] * 12)
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
           
    try:

        employment_df['Income_Duration'] = (employment_df['CIBIL_Report_Date'] -  employment_df['Date_Reported']).dt.days
        employment_df = employment_df.reset_index(drop=True)
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    


    return employment_df
