
import os
import numpy as np



DATASETS = [
    'header',
    'account',
    'address',
    'employment',
    'enquiry',
    'identification',
    'name',
    'phone',
    'score']


DEFAULT_LOAN_TYPE_DICT = {
    'creditcard': ['10','10.0'],
    'home': [ '2','2.0', '3','3.0', '02', '03','42','42.0'],
    'personal': ['5', '05', '41','5.0','41.0'],
     'gold': [ '7', '07','7.0'],
    'secured_business_loan' : ['40','40.0','50','50.0','51','51.0',
                               '52','52.0','53','53.0', '54.0','54','59','59.0'],
    'unsecured_business_loan' : ['61','61.0'],
    'loan_against_bank_deposits' : ['15','15.0'],
    'consumer' : ['6','06','6.0']}

DEFAULT_ACCOUNT_DPD_MAPPING =  {'STD': 0,
                        'SMA': 45,
                        'SUB': 91,
                        'DBT': 91,
                        'LSS': 91,
                        'XXX': -2,
                        'CCC': -1,
                        'YYY': -1}


DEFAULT_ACCOUNT_OWNERSHIP_MAPPING = {'Individual': ['01','1'],
                              'Non-Individual': ['02','2','03','3','04','4']}
                              


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
artefacts_folder = os.path.join(PROJECT_ROOT, 'artefacts')


DEFAULT_SCHEMA_DICT_PATH = os.path.join(
    artefacts_folder, 'DEFAULT_SCHEMA_DICT.pkl')

DEFAULT_FEATURES_LIST_PATH = os.path.join(
    artefacts_folder, 'DEFAULT_FEATURES_LIST.pkl')


DEFAULT_NUMERICAL_INTERESTING_DICT  =  {

    'enquiry': {
        'Enquiry_Amount' : [0,25000,np.inf]
    },
   
    "account": {
        "Current_Balance": [0, 50000, 200000, 500000, np.inf],
        "frac_Current_Balance__High_Credit_Sanctioned_Amount": [0, 0.3,0.8, np.inf], 
        "frac_Amount_Overdue__Current_Balance": [0, 0.3, np.inf], 
        "frac_Actual_Payment_Amount__EMI_Amount": [0, 0.3, 0.8, np.inf],
        "frac_Current_Balance__Value_of_Collateral" : [0, 0.1, 0.8, np.inf],
        "High_Credit_Sanctioned_Amount": [0, 50000, 200000, 500000, np.inf],
        "Loan_frac_Tenure": [0, 0.7, np.inf]}}


DEFAULT_CLUSTERING_NUMERICAL_INTERESTING_DICT =  {
    "enquiry": {
        "Cluster_Mean_Enquiry_Amount": [0, 1000, 100000, 500000, np.inf]},
   
    "account": {
        "Cluster_Intra_Cluster_Distance" : [0,5,10,15,np.inf],
        "Cluster_Amount_Deviation" : [0,0.05,0.1,0.5,1,np.inf],
        "Cluster_Mean_Enquiry_Amount": [0, 1000, 100000, 500000, np.inf]}
        }
 
error_message_str = 'Error Message'

