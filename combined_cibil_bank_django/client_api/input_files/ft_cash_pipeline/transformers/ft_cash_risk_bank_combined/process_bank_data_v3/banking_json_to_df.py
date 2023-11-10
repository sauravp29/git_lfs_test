import pandas as pd
import numpy as np
from .exceptions import *


def change_date(x):
    return '-'.join(x.split('/')[::-1])


def parse_perfios(data, application_id, c_date):
    try:
        txns = []
        for j in range(len(data)):
            if 'PERFIOS_DATA' in data[j].keys():
                txns.extend(data[j]['PERFIOS_DATA']['accountXns'][0]['xns'])
            elif 'perfiosResponse' in data[j].keys():
                txns.extend(data[j]['perfiosResponse']['accountXns'][0]['xns'])

        df_txn = pd.DataFrame(txns)
        df_txn['application_id'] = [application_id] * df_txn.shape[0]
        df_txn['application_id'] = df_txn['application_id'].astype('O')
        df_txn['application_created_date'] = [c_date] * df_txn.shape[0]
        df_txn = df_txn.rename(
            columns={
                'amount': 'amount__c',
                'balance': 'current_balance__c',
                'narration': 'description'})
        print(df_txn.columns)
        df_txn['tx_type__c'] = (df_txn['amount__c'] > 0).replace(
            {True: 'CREDIT', False: 'DEBIT'})
        df_txn['amount__c'] = np.abs(df_txn['amount__c'])
        df_txn = df_txn[['application_id',
                         'application_created_date',
                         'amount__c',
                         'current_balance__c',
                         'date',
                         'description',
                         'tx_type__c']]
        df_txn['ind'] = df_txn.index
        df_txn = df_txn.sort_values(by=['date', 'ind'])
        return df_txn

    except Exception:
        raise Exception('Unable to parse the perfios data')


def parse_qbera(json_data, application_id, c_date):

    try:
        bank_json = json_data
        if isinstance(bank_json['PIR:Data']['Account'], list):
            fin_json_lis = []
            for acct in range(len(bank_json['PIR:Data']['Account'])):
                fin_json_lis.extend(
                    bank_json['PIR:Data']['Account'][acct]['Xns']['Xn'])
            df_txn = pd.DataFrame(fin_json_lis)
        else:
            df_txn = pd.DataFrame(
                bank_json['PIR:Data']['Account']['Xns']['Xn'])

        df_txn['application_id'] = [application_id] * df_txn.shape[0]
        df_txn['application_id'] = df_txn['application_id'].astype('O')
        df_txn['application_created_date'] = [c_date] * df_txn.shape[0]
        df_txn = df_txn.rename(
            columns={
                '@amount': 'amount__c',
                '@balance': 'current_balance__c',
                '@narration': 'description',
                '@date': 'date'})
        df_txn['amount__c'] = df_txn['amount__c'].astype('float64')
        df_txn['tx_type__c'] = (df_txn['amount__c'] > 0).replace(
            {True: 'CREDIT', False: 'DEBIT'})
        df_txn['amount__c'] = np.abs(df_txn['amount__c'])
        df_txn = df_txn[['application_id',
                         'application_created_date',
                         'amount__c',
                         'current_balance__c',
                         'date',
                         'description',
                         'tx_type__c']]
        return df_txn
    except Exception as e:
        raise Exception('Unable to parse the qbera data')


def parse_fin360(data, application_id, c_date):
    try:
        if 'error_status' in data:
            raise Exception('Error response From Fin360')
        txns = data['transactions']
        df_txn = pd.DataFrame(txns)
        df_txn['application_id'] = [application_id] * df_txn.shape[0]
        df_txn['application_id'] = df_txn['application_id'].astype('O')
        df_txn['application_created_date'] = [c_date] * df_txn.shape[0]
        df_txn = df_txn.rename(
            columns={
                'amount': 'amount__c',
                'balanceAfterTransaction': 'current_balance__c',
                'dateTime': 'date',
                'type': 'tx_type__c'})
        df_txn['date'] = df_txn['date'].apply(change_date)
        df_txn['amount__c'] = np.abs(df_txn['amount__c'])
        df_txn = df_txn[['application_id',
                         'application_created_date',
                         'amount__c',
                         'current_balance__c',
                         'date',
                         'description',
                         'tx_type__c']]
        return df_txn
    except Exception:
        raise Exception('Unable to parse the fin360 data')


def parse_monsoon(data, application_id, c_date, deduce_type=False):
    try:
        txns = data['transactions']
        df_txn = pd.DataFrame(txns)
        df_txn['application_id'] = [application_id] * df_txn.shape[0]
        df_txn['application_created_date'] = [c_date] * df_txn.shape[0]
        df_txn['application_id'] = df_txn['application_id'].astype('O')
        df_txn = df_txn.rename(
            columns={
                'amount': 'amount__c',
                'balance': 'current_balance__c',
                'narration': 'description'})
        if deduce_type:
            df_txn['tx_type__c'] = (df_txn['amount__c'] > 0).replace(
                {True: 'CREDIT', False: 'DEBIT'})
        df_txn['amount__c'] = np.abs(df_txn['amount__c'])
        df_txn = df_txn[['application_id',
                         'application_created_date',
                         'amount__c',
                         'current_balance__c',
                         'date',
                         'description',
                         'tx_type__c']]
        df_txn['ind'] = df_txn.index
        df_txn = df_txn.sort_values(by=['date', 'ind'])
        return df_txn
    except Exception:
        raise Exception(
            f'Unable to parse the json data for app_id {application_id}')


def rename_dataframe_using_yaml(data, YAML_FILE, app_id, deduce_type=False):
    try:
        YAML_FILE = YAML_FILE['transactions']
        df_txn = data.copy(deep=True)
        # df_txn['application_id'] = [application_id]*df_txn.shape[0]
        application_id = YAML_FILE['application_id']
        app_id = df_txn[application_id].iloc[0]
        df_txn[application_id] = df_txn[application_id].astype('O')
        if deduce_type:
            df_txn = df_txn.rename(
                columns={
                    YAML_FILE['amount']: 'amount__c',
                    YAML_FILE['balance']: 'current_balance__c',
                    YAML_FILE['description']: 'description',
                    application_id: 'application_id',
                    YAML_FILE['date']: 'date'})
            df_txn['tx_type__c'] = (df_txn['amount__c'] > 0).replace(
                {True: 'CREDIT', False: 'DEBIT'})
        else:
            # pass
            df_txn = df_txn.rename(
                columns={
                    YAML_FILE['amount']: 'amount__c',
                    YAML_FILE['balance']: 'current_balance__c',
                    YAML_FILE['description']: 'description',
                    application_id: 'application_id',
                    YAML_FILE['date']: 'date',
                    YAML_FILE['type']: 'tx_type__c'})

        df_txn['amount__c'] = np.abs(df_txn['amount__c'])
        df_txn = df_txn[['application_id',
                         'application_created_date',
                         'amount__c',
                         'current_balance__c',
                         'date',
                         'description',
                         'tx_type__c']]
        df_txn['ind'] = df_txn.index
        df_txn = df_txn.sort_values(by=['date', 'ind'])
        return df_txn
    except Exception as e:
        raise DataFrameRenamingException(
            f'Unable to rename the dataframe for app_id {app_id}' + str(e))
