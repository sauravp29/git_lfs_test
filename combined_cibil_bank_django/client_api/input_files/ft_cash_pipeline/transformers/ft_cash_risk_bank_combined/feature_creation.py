import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
from process_bank_data_v3.banking_parser import BankingParser
import multiprocessing as mp
import traceback
from cibil_feature_creation_v3.cibil_parser_v3 import CibilFeatureCreation


def create_features_bank(header_df,bank_df):
    

    bank_df['name'] = bank_df['name'].replace({"Credit":'CREDIT','Debit':'DEBIT'})
    bank_df = bank_df[bank_df['transaction_date'] <= bank_df['createdOn']]
    bank_df = bank_df.reset_index(drop=True)
    bank_parser_object = BankingParser('yaml_files/ft_cash_bank.yaml',MAPPING_FILE=header_df)
    bank_features_df = bank_parser_object.process_bank_data_batch(bank_df,n_jobs = mp.cpu_count())
    
    bank_features_df_final = [file for file in bank_features_df if (isinstance(file,pd.Series) or isinstance(file,pd.DataFrame))]
    if len(bank_features_df_final)>0:
        bank_features_df_final = pd.concat(bank_features_df_final,axis=0,ignore_index=True)
        bank_features_df_final = bank_features_df_final.reset_index(drop=True)
    else:
        bank_features_df_final = pd.DataFrame()
    error = [file for file in bank_features_df if not (isinstance(file,pd.Series) or isinstance(file,pd.DataFrame))]

    bank_output_features=pd.read_pickle('pickled_objects/bank/bank_train_columns.pkl')
    bank_features_df_final=bank_features_df_final[bank_output_features]
    

    
    return bank_features_df_final,error



def create_features_cibil(header_df,account_df):

    input_dict={'header':header_df.copy(deep=True),'account':account_df.copy(deep=True)}
    
    rename_and_dtype_dict = {'account': 
               {'Account_Number': ('Account_Number','string'),
  'Account_Type': ('Account_Type','string'),
  'Actual_Payment_Amount': ('Actual_Payment_Amount','numeric'),
  'Amount_Overdue': ('Amount_Overdue','numeric'),
  'CIBIL_Remarks_Code': ('CIBIL_Remarks_Code','string'),
  'Cash_Limit': ('Cash_Limit','numeric'),
  'Credit_Limit': ('Credit_Limit','numeric'),
  'Current_Balance': ('Current_Balance','numeric'),
  'Date_Closed': ('Date_Closed','datetime'),
  'Date_Opened_Disbursed': ('Date_Opened_Disbursed','datetime'),
  'Date_Reported_and_Certified': ('Date_Reported_and_Certified','datetime'),
  'Date_of_Entry_for_CIBIL_Remarks_Code': ('Date_of_Entry_for_CIBIL_Remarks_Code','datetime'),
  'Date_of_Entry_for_Error_Code': ('Date_of_Entry_for_Error_Code','datetime'),
  'Date_of_Last_Payment': ('Date_of_Last_Payment','datetime'),
  'EMI_Amount': ('EMI_Amount','numeric'),
  'Error_Code': ('Error_Code','string'),
  'High_Credit_Sanctioned_Amount': ('High_Credit_Sanctioned_Amount','numeric'),
  'Ownership_Indicator': ('Ownership_Indicator','string'),
  'Payment_Frequency': ('Payment_Frequency','numeric'),
  'Payment_History_1': ('Payment_History_1','string'),
  'Payment_History_2': ('Payment_History_2','string'),
  'Payment_History_End_Date': ('Payment_History_End_Date','datetime'),
  'Payment_History_Start_Date': ('Payment_History_Start_Date','datetime'),
  'Rate_Of_Interest': ('Rate_Of_Interest','numeric'),
  'Repayment_Tenure': ('Repayment_Tenure','numeric'),
  'Reporting_Member_Short_Name': ('Reporting_Member_Short_Name','string'),
  'Segment_Tag': ('Segment_Tag','string'),
  'Settlement_Amount': ('Settlement_Amount','numeric'),
  'Suit_Filed_Wilful_Default': ('Suit_Filed_Wilful_Default','string'),
  'Type_of_Collateral': ('Type_of_Collateral','string'),
  'Value_of_Collateral': ('Value_of_Collateral','numeric'),
  'Written_off_Amount_Principal': ('Written_off_Amount_Principal','numeric'),
  'Written_off_Amount_Total': ('Written_off_Amount_Total','numeric'),
  'Written_off_and_Settled_Status': ('Written_off_and_Settled_Status','numeric')},
        
 'header': {'enquiryMemberUserID': ('enquiryMemberUserID','string'),
  'memberReferenceNumber': ('memberReferenceNumber','string'),
  'serialversionuid': ('serialversionuid','string'),
  'subjectReturnCode': ('subjectReturnCode','string'),
  'timeProceed': ('timeProceed','datetime'),
  'version': ('version','string'),
  'enquiryControlNumber': ('enquiryControlNumber','string'),
  'CIBIL_Report_Date': ('CIBIL_Report_Date','datetime')}}
    
  
    selected_features_list = CibilFeatureCreation.get_default_features(['account'])
    cibil_output_features=pd.read_pickle('pickled_objects/cibil/cibil_train_columns.pkl')
    selected_features_list=[x for x in selected_features_list if x in cibil_output_features]
    SCHEMA_DICT = CibilFeatureCreation.create_schema_dict(selected_features_list)
    object_cibil_parser = CibilFeatureCreation(RENAME_AND_DTYPE_DICT=rename_and_dtype_dict,
                                   default_date_format='%Y-%m-%d',
                                    SCHEMA_DICT=SCHEMA_DICT)
    cibil_features,error = object_cibil_parser.process_cibil_data_batch(input_dict,n_jobs=-1,chunk_size=1000,
                                                                        feature_list=selected_features_list)
    if len(error)>0:
        error=[error]

        
    return cibil_features,error