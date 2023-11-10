from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from ..utils.helper_functions import make_numerical_interesting_vars

def process_tl_mom_df(account_df,tl_mom_used_cols):

    print('processing tl_mom data')


    # Make tl_mom (dpd_string columns -> dpd rows month wise)
    i = [i[j - 3:j] for i in account_df['con_pmt_history']
         for j in reversed(range(3, len(i) + 3, 3))]

    # tl_mom_cols = [
    #     'CIBIL_Report_Date',
    #     'Date_Closed',
    #     'Current_Balance',
    #     'Account_Type',
    #     'Repayment_Tenure',
    #     'enquiryControlNumber',
    #     'Date_Opened_Disbursed',
    #     'Date_Reported_and_Certified_Payment_History_End_Date_duration',
    #     'Actual_Payment_Amount',
    #     "Written_off_and_Settled_Status_cleaned",
    #     "Current_Balance_categorical",
    #     'Loan_Status',
    #     'frac_Current_Balance_High_Credit_Sanctioned_Amount_categorical',
    #     'tl_u_id']


    tl_mom_cols = ['con_pmt_history',
                   'Payment_History_End_Date',
                   'tl_u_id',
                   'CIBIL_Report_Date',
                   'Account_Type_og',
                   'Date_Closed',
                   'Repayment_Tenure',
                   'enquiryControlNumber']

    common_cols = set(tl_mom_cols).union(tl_mom_used_cols).intersection(set(account_df.columns))
    common_cols = list(common_cols)

    tl_mom_df = account_df[common_cols].copy(deep=True)
   
    j = tl_mom_df[common_cols].values.repeat(
        tl_mom_df['con_pmt_history'].str.len() / 3, axis=0)

    pmt_dates = [i + relativedelta(months=j)
                 for n, i in enumerate(tl_mom_df['Payment_History_End_Date'])
                 for j in range(int(len(tl_mom_df['con_pmt_history'].iloc[n]) / 3))]

    tl_mom_df = pd.DataFrame(j, columns=tl_mom_df.columns)
    tl_mom_df = tl_mom_df.assign(**{'dpd': i, 'Pmt_Date': pmt_dates})
    tl_mom_df['tl_u_id'] = tl_mom_df['tl_u_id'].astype(int)

    if 'CIBIL_Report_Date' in tl_mom_df.columns:
        tl_mom_df['CIBIL_Report_Date'] = pd.to_datetime(
            tl_mom_df['CIBIL_Report_Date'], format='%Y-%m-%d')
        tl_mom_df['Pmt_Date'] = pd.to_datetime(
            tl_mom_df['Pmt_Date'], format='%Y-%m-%d')

        tl_mom_df['CIBIL_Report_Date_Pmt_Date_duration'] = (
            tl_mom_df['CIBIL_Report_Date'] - tl_mom_df['Pmt_Date']).dt.days
    

    tl_mom_df['emi_u_id'] = tl_mom_df.index

    # Computing Feature : loan_status_mom
    tl_mom_df['loan_status_mom'] = True

    # Different logic for evergreen tradelines, Check Revolving loan acc
    # (cibil guide appendix)
    always_open_acc = [10.0, 31.0, 35.0, 36.0, 16.0, 12.0, 37.0, 38.0]
    always_open_acc_mask = tl_mom_df['Account_Type_og'].isin(always_open_acc)

    date_closed_mask = (
        tl_mom_df[always_open_acc_mask]["Date_Closed"] < tl_mom_df[always_open_acc_mask]["Pmt_Date"]) & (
        tl_mom_df[always_open_acc_mask]["Date_Closed"].notnull())
    tl_mom_df['loan_status_mom'][always_open_acc_mask] = date_closed_mask

    # Other tradelines
    tl_mom_df['loan_status_mom'][~always_open_acc_mask] = tl_mom_df["dpd"][~always_open_acc_mask] != "CCC"

    tl_mom_df['is_open'] = tl_mom_df.groupby(
        "tl_u_id")["loan_status_mom"].transform('all')
    tl_mom_df['loan_status_mom'] = tl_mom_df['loan_status_mom'].replace(
        {True: "open", False: "closed"})

    # Computing Feature : dpd_numerical
    # In order to have a numercial feature with the dpd value, we need to convert the categorical variable in dpd string to
    # numerical values
    DPD_VAR_NUM_MAP = {
        "STD": "000",
        "SMA": "031",
        "LSS": "121",
        "SUB": "091",
        "DBT": "091",
        "CCC": "000",
        "YYY": np.nan}
    tl_mom_df["dpd_numerical"] = tl_mom_df['dpd'].replace(
        DPD_VAR_NUM_MAP, regex=True)

    # YYY should be mapped to last dpd status
    tl_mom_df["dpd_numerical"] = tl_mom_df.groupby(
        "tl_u_id")["dpd_numerical"].ffill()

    # Required for interpolation to work
    tl_mom_df["dpd_numerical"] = tl_mom_df["dpd_numerical"].replace(
        'XXX', np.nan)

    # string to float
    tl_mom_df["dpd_numerical"] = tl_mom_df["dpd_numerical"].astype(float)

    # Apply interpolation, tl_u_id as a index, and both direction
    tl_mom_df["dpd_numerical"] = tl_mom_df.groupby('tl_u_id')["dpd_numerical"].apply(
        lambda account: account.interpolate(method='index', limit_direction="both"))

    # DPD string with only "XXX" remains with nan value. Replacing with '0'
    tl_mom_df["dpd_numerical"] = tl_mom_df["dpd_numerical"].fillna(0)
    tl_mom_df["dpd_numerical"] = tl_mom_df["dpd_numerical"].astype(int)

    # Computing Feature : frac_paid_month_Repayment_Tenure
    # All the mask that will be used
    closed_loans = ~tl_mom_df['is_open']
    missed_pay = tl_mom_df['dpd_numerical'] > 0
    closed_mask = (tl_mom_df["loan_status_mom"] == "closed")
    open_padded_mask = (tl_mom_df["dpd"] == "YYY")

    tl_mom_df["paid_month"] = 1
    tl_mom_df["paid_month"][missed_pay | open_padded_mask] = np.nan
    tl_mom_df["paid_month"][~missed_pay] = tl_mom_df[~missed_pay].groupby(
        'tl_u_id').cumcount() + 1
    # Replacing nan with zero or last positive val
    tl_mom_df["paid_month"] = tl_mom_df.groupby(
        "tl_u_id")["paid_month"].ffill().fillna(0).astype('int32')
    tl_mom_df["paid_month"][closed_mask] = tl_mom_df['Repayment_Tenure'][closed_mask]
    # Calculate frac_paid_month_Repayment_Tenure
    tl_mom_df["paid_month"] = tl_mom_df["paid_month"].astype(float)
    tl_mom_df['Repayment_Tenure'] = tl_mom_df['Repayment_Tenure'].astype(float)
    tl_mom_df["frac_paid_month_Repayment_Tenure"] = tl_mom_df["paid_month"] / \
        tl_mom_df['Repayment_Tenure']
    

    tl_mom_df = make_numerical_interesting_vars(tl_mom_df,{
        "frac_paid_month_Repayment_Tenure": [0, 0.5, 0.8, np.inf],
    })

    # Making bulk payment in last month before closing account
    tl_mom_df["future_ratio"] = tl_mom_df.groupby(
        'tl_u_id')['frac_paid_month_Repayment_Tenure'].shift(-1)
    tl_mom_df["future_ratio"][tl_mom_df["future_ratio"] != 1] = np.nan
    tl_mom_df["frac_paid_month_Repayment_Tenure"][tl_mom_df["future_ratio"].notnull()] = 1

    # Computing Feature : dpd_categorical
    isnumeric_mask = tl_mom_df['dpd'].astype(str).str.isdigit()

    conditions = [
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 0) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 5)),
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 5) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 10)),
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 10) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 20)),
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 20) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 30)),
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 30) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 45)),
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 45) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 60)),
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 60) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 90)),
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 90) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 120)),
        ((tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 120) & (tl_mom_df['dpd'][isnumeric_mask].astype(int) < 240)),
        (tl_mom_df['dpd'][isnumeric_mask].astype(int) >= 240),
    ]

    choices = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
    tl_mom_df['dpd'][isnumeric_mask] = np.select(conditions, choices)
    tl_mom_df = tl_mom_df.rename(columns={"dpd": "dpd_categorical"})


    return tl_mom_df
