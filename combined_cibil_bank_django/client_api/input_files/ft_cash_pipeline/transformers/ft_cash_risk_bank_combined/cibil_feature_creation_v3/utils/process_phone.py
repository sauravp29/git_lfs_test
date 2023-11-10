import warnings

def process_phone_df(phone_df):

    print('processing phone data')

    try:

        if 'Phone_Type' in phone_df.columns:
            phone_df['Mobile_Phone_Indicator'] = phone_df['Phone_Type'].isin(['01','1'])
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    phone_df.index.rename('Telephone_Index', inplace=True)
    phone_df = phone_df.reset_index(drop=False)

    return phone_df
