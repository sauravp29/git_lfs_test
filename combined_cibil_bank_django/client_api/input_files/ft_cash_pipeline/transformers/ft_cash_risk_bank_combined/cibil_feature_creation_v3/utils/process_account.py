import copy
import numpy as np
from .helper_functions import create_ratios,make_categorical_interesting_vars
import pandas as pd
import warnings


def process_account_df(account_df, header_df,loan_type_dict,
                       account_dpd_mapping,ownership_type_dict):
    print('processing accounts data')


    try:
        #adding CIBIL_Report_Date
        account_df = account_df.merge(header_df, on='enquiryControlNumber',
                                    how='inner',suffixes=('','_y'))
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:

    # if Payment_History_Start_Date is missing impute with
    # Date_Reported_and_Certified
        if ('Payment_History_Start_Date' in account_df.columns) and (
                'Date_Reported_and_Certified' in account_df.columns):
            account_df['Payment_History_Start_Date'] = account_df['Payment_History_Start_Date'].where(
                account_df['Payment_History_Start_Date'].notnull(), account_df['Date_Reported_and_Certified'])
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:

    # if Payment_History_Start_Date is missing impute with CIBIL_Report_Date
        if ('Payment_History_Start_Date' in account_df.columns) and (
                'CIBIL_Report_Date' in account_df.columns):
            account_df['Payment_History_Start_Date'] =\
                account_df['Payment_History_Start_Date']\
                .where(account_df['Payment_History_Start_Date'].notnull(),
                    account_df['CIBIL_Report_Date'])
            
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:
    # if Payment_History_End_Date is missing impute with Date_Opened_Disbursed
        if ('Payment_History_End_Date' in account_df.columns) & (
                'Date_Opened_Disbursed' in account_df.columns):
            account_df['Payment_History_End_Date'] =\
                account_df['Payment_History_End_Date']\
                .where(account_df['Payment_History_End_Date'].notnull(),
                    account_df['Date_Opened_Disbursed'])
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:
    # if Payment_History_End_Date or Payment_History_Start_Date are null, drop
    # from the dataset
        if ('Payment_History_End_Date' in account_df.columns) & (
                'Payment_History_Start_Date' in account_df.columns):
            drop_accs = account_df['Payment_History_End_Date'].notnull(
            ) & account_df['Payment_History_Start_Date'].notnull()
            account_df = account_df.loc[drop_accs, :]
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    
    try:
        if 'Date_Opened_Disbursed' in account_df.columns:
            account_df['Date_Opened_Disbursed'] = account_df['Date_Opened_Disbursed'].where(
                account_df['Date_Opened_Disbursed'].notnull(), account_df['Payment_History_End_Date'])
            
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
        


    #TODO: check this with Nitin
    # # if date opened is after payment history end date, set it to payment history end date
    # if 'Date_Opened_Disbursed' in account_df.columns and 'Payment_History_End_Date' in account_df:
    #     account_df['Date_Opened_Disbursed'] = np.where(account_df['Date_Opened_Disbursed'] > account_df['Payment_History_End_Date'],
    #                                                    account_df['Payment_History_End_Date'],account_df['Date_Opened_Disbursed'])
    
    try:
        # if date closed is before 1980, set it to NaT because it is probably a
        # typo/fill value
        if 'Date_Closed' in account_df.columns:
            date_old_mask = account_df['Date_Closed'].dt.year >= 1980
            account_df['Date_Closed'] = account_df['Date_Closed'].where(date_old_mask,pd.NaT)
        
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
            
    try:
        if "Date_Reported_and_Certified" in account_df.columns:
            account_df['Date_Reported_and_Certified'] = account_df['Date_Reported_and_Certified'] \
                .fillna(account_df['Payment_History_Start_Date'])
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    
    
    try:
        account_df = account_df.sort_values(by='Date_Opened_Disbursed', ascending=True,kind='mergesort')
        account_df = account_df.reset_index(drop=True)
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
#   
    try:
        account_df = impute_open_loans(account_df)
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    try:
        account_df = impute_closed_loans(account_df)
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    
    try:
        if 'Payment_History_1' in account_df.columns:
            account_df['Payment_History_1'] = account_df['Payment_History_1'].fillna('')
            account_df['Payment_History_1'] = account_df['Payment_History_1'].str.replace(' ','')
            account_df['Payment_History_1'] = account_df['Payment_History_1'].str.replace('nan','')
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    

    try:
        if 'Payment_History_2' in account_df.columns:
            account_df['Payment_History_2'] = account_df['Payment_History_2'].fillna('')
            account_df['Payment_History_2'] = account_df['Payment_History_2'].str.replace(' ','')
            account_df['Payment_History_2'] = account_df['Payment_History_2'].str.replace('nan','')
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    


    
    try:
        if 'con_pmt_history' not in account_df.columns:

            if ('Payment_History_0' in account_df.columns) & \
                ('Payment_History_1' in account_df.columns) & \
                    ('Payment_History_2' in account_df.columns):
                account_df['con_pmt_history'] =\
                    account_df['Payment_History_0'].astype(str).replace('nan', '') + \
                    account_df['Payment_History_1'].astype(str).replace('nan', '') \
                    + account_df['Payment_History_2'].astype(str).replace('nan', '')


        else:
            if ('Payment_History_0' in account_df.columns):
                account_df['con_pmt_history'] =\
                    account_df['Payment_History_0'].astype(str).replace('nan', '') + \
                    account_df['con_pmt_history'].astype(str).replace('nan', '')
        
        account_df['con_pmt_history'] = account_df['con_pmt_history'].str.upper()
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    

    try:
        if ('Repayment_Tenure' in account_df.columns) and (
                'Payment_Frequency' in account_df.columns):

            c = account_df['Payment_Frequency'].astype('str').replace(
                {'1.0': 4, '2.0': 2, '3.0': 1, '4.0': 0.25, 'nan': np.nan})
            c = c.astype(float)
            account_df['Number_of_payment'] = account_df['Repayment_Tenure'] * c
            account_df['Number_of_payment'] = account_df['Number_of_payment'].where(
                account_df['Number_of_payment'].notnull(), account_df['Repayment_Tenure'])
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    
    try:
        if "Account_Type" in account_df.columns:

            account_df['Account_Type_og'] = account_df['Account_Type']
            account_df['Account_Type_business_personal'] = account_df['Account_Type']

            all_acc = set(account_df['Account_Type'].unique().tolist())
            # Evergreen account
            evergreen_acc = set(['10', '31', '35', '36', '16', '12', '37', '38'])
            others_acc = all_acc - evergreen_acc
            business_acc = set(['9',
                                '17',
                                '33',
                                '35',
                                '39',
                                '40',
                                '50',
                                '51',
                                '52',
                                '53',
                                '54',
                                '55',
                                '56',
                                '57',
                                '58',
                                '59',
                                '60',
                                '61'])
            personal_acc = set(['1',
                                '2',
                                '3',
                                '4',
                                '5',
                                '10',
                                '11',
                                '12',
                                '13',
                                '15',
                                '16',
                                '31',
                                '32',
                                '34',
                                '36',
                                '37',
                                '38',
                                '44'])
            others_acc_bp = (all_acc - business_acc) - personal_acc
            col_dict = {
                "Account_Type": {
                    "evergreen": list(evergreen_acc),
                    "others": list(others_acc)},
                "Account_Type_business_personal": {
                    "Business": list(business_acc),
                    "Personal": list(personal_acc),
                    "others": list(others_acc_bp)} }

            account_df = make_categorical_interesting_vars(account_df, col_dict)

           
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    
    try:
        loan_type_replace_dict = {}

        for key,value in loan_type_dict.items():
            for val in value:
                loan_type_replace_dict[val] = key
      
        account_df['Account_Type_Cleaned'] = account_df['Account_Type'].str.strip().map(loan_type_replace_dict)
        account_df['Account_Type_Cleaned'] = account_df['Account_Type_Cleaned'].fillna('others')
        
        account_df['Account_Type'] = account_df['Account_Type'].str.strip().replace(loan_type_replace_dict)
        
        
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

   
    
    try:
        account_df['Settlement_Amount_present'] = account_df['Settlement_Amount'].notna()
        account_df['Settlement_Amount_present'] = account_df['Settlement_Amount_present'].astype(str)
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
        

    try:
        account_df['Loan_Tenure'] = (account_df['CIBIL_Report_Date'] - account_df['Date_Opened_Disbursed']).dt.days
        account_df['Loan_frac_Tenure'] = (account_df['Loan_Tenure']/account_df['Repayment_Tenure']*12*30)
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:
        account_df['Type_of_Collateral'] = account_df['Type_of_Collateral'].replace({'nan':'nan','0.0' : 'no collateral',
                                                                                    '1.0' : 'collateral', '2.0' : 'collateral',
                                                                                    '3.0' : 'collateral','4.0' : 'collateral'})
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))



    try:
            
        ratio_tuple_list = [('Amount_Overdue','Current_Balance'),
                            ('Current_Balance','Value_of_Collateral'),
                            ('Current_Balance','High_Credit_Sanctioned_Amount'),
                            ('Actual_Payment_Amount','EMI_Amount')]
        

        account_df = create_ratios(account_df,ratio_tuple_list)
        account_df['Perc_Paid_Off_wrt_High_Credit_Sanctioned_Amount'] = 100*(1 - account_df['frac_Current_Balance__High_Credit_Sanctioned_Amount'])
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    

    try:
        ownership_type_replace_dict = {}

        for key,value in ownership_type_dict.items():
            for val in value:
                ownership_type_replace_dict[val] = key


        account_df['Guarantor_Indicator'] = account_df['Ownership_Indicator'].isin(['03','3'])
        account_df['Ownership_Indicator_cleaned'] = account_df['Ownership_Indicator'].str.strip().replace(ownership_type_replace_dict)

    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    
    
    try:
        account_df = build_payment_history_features(account_df,account_dpd_mapping)
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    
    try:
        account_df['Tenure'] = (account_df['Payment_History_Start_Date'] - account_df['Payment_History_End_Date']).dt.days
        account_df['Days_Since_Acc_Opened'] = (account_df['CIBIL_Report_Date'] - account_df['Date_Opened_Disbursed']).dt.days
        account_df['Days_Since_Acc_Closed'] = (account_df['CIBIL_Report_Date'] - account_df['Date_Closed']).dt.days
        # account_df['Closed_Loan_Flag'] = (account_df['Date_Closed'].notnull() | account_df['Current_Balance'] < 100)

    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    

    try:

        credit_card_mask = account_df['Account_Type'] == 'creditcard'
        account_df['High_Credit'] = account_df['High_Credit_Sanctioned_Amount'].where(credit_card_mask,np.nan)
        account_df['Sanctioned_Amount'] = account_df['High_Credit_Sanctioned_Amount'].where(~credit_card_mask,np.nan)

        account_df.loc[credit_card_mask, 'Perc_Paid_Off_wrt_Credit_Limit_and_Sanctioned_Amount'] = 100 * \
        (1 - (account_df.loc[credit_card_mask, 'Current_Balance'] / account_df.loc[credit_card_mask, 'Credit_Limit']))

        account_df.loc[~credit_card_mask, 'Perc_Paid_Off_wrt_Credit_Limit_and_Sanctioned_Amount'] = 100 * \
            (1 - (account_df.loc[~credit_card_mask, 'Current_Balance'] / account_df.loc[~credit_card_mask, 'Sanctioned_Amount']))
    

    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    
    #TODO: DFS preprocessing 
    try:
        account_df['Date_Reported_and_Certified_Payment_History_End_Date_duration'] = (account_df['Date_Reported_and_Certified'] - 
                                                                                       account_df['Payment_History_End_Date']).dt.days
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))
    

    try:
        account_df['Date_Reported_and_Certified_Date_of_Last_Payment_duration'] = (account_df['Date_Reported_and_Certified'] - 
                                                                                       account_df['Date_of_Last_Payment']).dt.days
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:
        account_df['Date_Reported_and_Certified_Payment_History_Start_Date_duration'] = (account_df['Date_Reported_and_Certified'] - 
                                                                                       account_df['Payment_History_Start_Date']).dt.days
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))



    try:
        account_df['Date_Reported_and_Certified_Payment_History_End_Date_duration'] = (account_df['Date_Reported_and_Certified'] - 
                                                                                       account_df['Payment_History_End_Date']).dt.days
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))


    try:
        account_df['Written_off_and_Settled_Status_cleaned'] = account_df['Written_off_and_Settled_Status'].astype(str)
                                                                                       
    
    except Exception as e:
        # raise e
        print(str(e))
        warnings.warn(str(e))

    account_df.index.rename('tl_u_id', inplace=True)
    account_df = account_df.reset_index(drop=False)

    return account_df


def impute_open_loans(df):
    """Impute the difference (in months) between Date_Reported_and_Certified and Payment History Start Date,for open loans, with YYY"""

    if 'Loan_Status' not in df.columns:
        df['Loan_Status']= df['Date_Closed'].notna()
        df['Loan_Status'] = df['Loan_Status'].replace({True:"closed",False:"open"})

    df_open=copy.deepcopy(df[df['Loan_Status']=="open"])

    df_open['Date_Reported_and_Certified'] = df_open['Date_Reported_and_Certified'].values.astype('datetime64[M]')


    df_open['Payment_History_0'] = ((df_open['Date_Reported_and_Certified'] -\
        df_open['Payment_History_Start_Date']) / np.timedelta64(1, 'M'))



    df_open['Payment_History_0'] = df_open['Payment_History_0'].apply(lambda x: 'YYY' * round(x))
    idx = df_open.index
    df.loc[idx, 'Payment_History_0'] = df_open.loc[idx, 'Payment_History_0']

    return df


def impute_closed_loans(df):
    """Impute the difference (in months) between Date_Reported_and_Certified and Payment History Start Date,for closed loans, with CCC"""

    df_closed=copy.deepcopy(df[df['Loan_Status']=="closed"])

    df_closed['Date_Reported_and_Certified'] = df_closed['Date_Reported_and_Certified'].values.astype('datetime64[M]')
    
    df_closed['Payment_History_00'] =\
        ((df_closed['Date_Reported_and_Certified'] -\
        df_closed['Payment_History_Start_Date']) / np.timedelta64(1, 'M'))

    df_closed['Payment_History_00'] = df_closed['Payment_History_00'].apply(lambda x: 'CCC' * round(x))
    idx = df_closed.index
    df.loc[idx, 'Payment_History_0'] = df_closed.loc[idx, 'Payment_History_00']
    return df



def process_dpd_string(s, accounts_dpd_mapping):
    l = []
    for i in range(0, len(s), 3):
        x = s[i:i + 3]
        try:
            x = int(x)
            l.append(x)
        except BaseException:
            x = accounts_dpd_mapping.get(x, np.nan)
            l.append(x)

    l = l[:36]  # taking only last 3 years
    # l = pd.Series(l).interpolate(method='linear',
    # limit_direction='both').tolist() # interpolate missing values

    if len(l) < 36:  # if less than 3 years data is available, pad with np.nan
        l = l + [np.nan] * (36 - len(l))
    return l





def build_payment_history_features(account_df,accounts_dpd_mapping):
    
    account_df = account_df.reset_index(drop=True)
    account_df['Payment_History'] = account_df['con_pmt_history'].apply(
        lambda x: process_dpd_string(x, accounts_dpd_mapping))
    
    payment_history_df = pd.DataFrame(
        account_df['Payment_History'].tolist(), columns=[
            f'phist_{i}' for i in range(
                1, 37)]).reset_index(
        drop=True)
    
    
    for m, prefix in zip([36, 24, 12, 6], [
                         '', '_in_the_last_24_months_of_Loan', '_in_the_last_12_months_of_Loan', '_in_the_last_6_months_of_Loan']):
        

        phist_cols = [f'phist_{i}' for i in range(1, m + 1)]

        # null means time period is before loan was disburse
        account_df['Count_of_NA_DPD' + prefix] = (payment_history_df[phist_cols].isna()).sum(axis=1)
        # means loan reporting is stopped
        account_df['Count_of_CCC_DPD'+ prefix] = (payment_history_df[phist_cols] == -1).sum(axis=1)
        # means dpd is unknown
        account_df['Count_of_XXX_DPD'+ prefix] = (payment_history_df[phist_cols] == -2).sum(axis=1)
        account_df['Count_of_0_DPD'+ prefix] = (payment_history_df[phist_cols] == 0).sum(axis=1)
        for d in [0, 15, 30, 45, 60, 90]:
            account_df[f'Count_of_{d}+_DPD'+ prefix] = (payment_history_df[phist_cols] > d).sum(axis=1)


        account_df[f'DPD_0+'+ prefix] = account_df['Count_of_0+_DPD'+ prefix] > 0
        account_df[f'DPD_0+'+ prefix] = account_df[f'DPD_0+'+ prefix].astype(str)

        account_df['Max_DPD'+ prefix] = payment_history_df[phist_cols].max(axis=1)
        account_df['Mean_DPD'+ prefix] = payment_history_df[phist_cols].mean(axis=1)
        account_df['Std_DPD'+ prefix] = payment_history_df[phist_cols].std(axis=1)
        account_df['Skew_DPD'+ prefix] = payment_history_df[phist_cols].skew(axis=1)

    return account_df
