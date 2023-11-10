import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pandas as pd
import numpy as np
import warnings
from feature_creation import create_features_bank,create_features_cibil
from utils import *
warnings.filterwarnings("ignore")


date_format = '%Y-%m-%d'
    

def transform(json_obj):
    try:
        try:
            identifier = json_obj['identifier']
            if identifier not in ['bank','cibil']:
                return generate_error_dict(95, ERROR_CODE_95, 400, 'Invalid Identifier', 'NA')
        except Exception as e:
            return generate_error_dict(95, ERROR_CODE_95, 400, str(e), 'NA')
        
        try:
            header_dict = json_obj['header_dict']
            if identifier=='bank':
                application_id = json_obj['header_dict']['loanApplicationId']
            elif identifier=='cibil':
                application_id = json_obj['header_dict']['enquiryControlNumber']
        except:
            return generate_error_dict(95, ERROR_CODE_95, 400, APP_ID_ERROR, 'NA')
            
        try:
            data_dict = json_obj['data_dict']
        except:
            return generate_error_dict(95, ERROR_CODE_95, 400, DATA_ERROR, application_id)
        
        try:
            header_df=pd.DataFrame(header_dict)
            if identifier=='bank':
                header_df['createdOn'] = pd.to_datetime(header_df['createdOn'],format=date_format)
            elif identifier=='cibil':
                header_df['CIBIL_Report_Date'] = pd.to_datetime(header_df['CIBIL_Report_Date'],format=date_format)
                
                
            data_df=pd.DataFrame(data_dict)
            if identifier=='bank':
                data_df['transaction_date'] = pd.to_datetime(data_df['transaction_date'],format=date_format)
                data_df['createdOn'] = pd.to_datetime(data_df['createdOn'],format=date_format)
                
                data_df['value'] = data_df['value'].astype('float64')
                data_df['balance'] = data_df['balance'].astype('float64')
                
            elif identifier=='cibil':
                data_df['Date_Opened_Disbursed'] = pd.to_datetime(data_df['Date_Opened_Disbursed'],format=date_format)
                data_df['Date_Closed'] = pd.to_datetime(data_df['Date_Closed'],format=date_format)
                data_df['Date_of_Last_Payment'] = pd.to_datetime(data_df['Date_of_Last_Payment'],format=date_format)
                data_df['Payment_History_Start_Date'] = pd.to_datetime(data_df['Payment_History_Start_Date'],format=date_format)
                data_df['Payment_History_End_Date'] = pd.to_datetime(data_df['Payment_History_End_Date'],format=date_format)
                data_df['Date_Reported_and_Certified'] = pd.to_datetime(data_df['Date_Reported_and_Certified'],format=date_format)
                
                data_df['High_Credit_Sanctioned_Amount'] = data_df['High_Credit_Sanctioned_Amount'].astype('float64')
                data_df['Repayment_Tenure'] = data_df['Repayment_Tenure'].astype('float64')
                data_df['Rate_Of_Interest'] = data_df['Rate_Of_Interest'].astype('float64')
                data_df['Credit_Limit'] = data_df['Credit_Limit'].astype('float64')
                data_df['Value_of_Collateral'] = data_df['Value_of_Collateral'].astype('float64')
                data_df['EMI_Amount'] = data_df['EMI_Amount'].astype('float64')
    
        except Exception as e:
            return generate_error_dict(98, ERROR_CODE_98, 400, str(e), application_id)
        
        try:
            if identifier=='bank':
                check_bank_data=pickle.load(open('pickled_objects/bank/bank_data_check.pkl', "rb"))
                check_bank_header=pickle.load(open('pickled_objects/bank/bank_header_check.pkl', "rb"))
                if (check_bank_data!=dict(data_df.dtypes)) or (check_bank_header!=dict(header_df.dtypes)):
                    return generate_error_dict(98, ERROR_CODE_98, 400, "Invalid column names or column datatype", application_id)
                    
            elif identifier=='cibil':
                check_cibil_data=pickle.load(open('pickled_objects/cibil/cibil_data_check.pkl', "rb"))
                check_cibil_header=pickle.load(open('pickled_objects/cibil/cibil_header_check.pkl', "rb"))
                if (check_cibil_data!=dict(data_df.dtypes)) or (check_cibil_header!=dict(header_df.dtypes)):
                    return generate_error_dict(98, ERROR_CODE_98, 400, "Invalid column names or column datatype", application_id)
        
        except Exception as e:
            return generate_error_dict(98, ERROR_CODE_98, 400, str(e), application_id)
            
            
        
        try:
            if identifier=='bank':
                feature_df,error = create_features_bank(header_df,data_df)
            elif identifier=='cibil':
                feature_df,error = create_features_cibil(header_df,data_df)
            
            if len(error)>0:
                return generate_error_dict(99, ERROR_CODE_99, 400, error[0], application_id)
                
        except Exception as e:
            return generate_error_dict(99, ERROR_CODE_99, 400, str(e), application_id)

        return feature_df
    
    except Exception as e:
        return generate_error_dict(93, ERROR_CODE_93, 400, str(e), application_id)