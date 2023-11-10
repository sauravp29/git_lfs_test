
import pandas as pd
from scipy import stats
import numpy as np
from datetime import datetime, timedelta
from functools import partial
import warnings


def calculate_mean_trend(x):
    return x.diff().mean()


def calculate_stdtrend(x):
    return x.diff().std()


def entropy_agg(s):
    distribution = s.value_counts(normalize=True,dropna=False)
    return stats.entropy(distribution)


def convert_datetime_to_floats(x):
    first = int(x.iloc[0].value * 1e-9)
    x = pd.to_numeric(x).astype(np.float64).values
    dividend = find_dividend_by_unit(first)
    x *= (1e-9 / dividend)
    return x


def convert_timedelta_to_floats(x):
    first = int(x.iloc[0].total_seconds())
    dividend = find_dividend_by_unit(first)
    x = pd.TimedeltaIndex(x).total_seconds().astype(np.float64) / dividend
    return x


def find_dividend_by_unit(time):
    """Finds whether time best corresponds to a value in
    days, hours, minutes, or seconds.
    """
    for dividend in [86400, 3600, 60]:
        div = time / dividend
        if round(div) == div:
            return dividend
    return 1


def calculate_trend(df_,date_col,primitive_col):
    df = df_.copy(deep=True)
    df = df.dropna(subset=[date_col,primitive_col])
    if df.shape[0] <= 2:
        return np.nan
    
    if isinstance(df[date_col].iloc[0], (datetime, pd.Timestamp)):
        x = convert_datetime_to_floats(df[date_col])
    else:
        x = df[date_col].values
      

    if isinstance(df[primitive_col].iloc[0], (datetime, pd.Timestamp)):
        y = convert_datetime_to_floats(df[primitive_col])
    elif isinstance(df[primitive_col].iloc[0], (timedelta, pd.Timedelta)):
        y = convert_timedelta_to_floats(df[primitive_col])
    else:
        y = df[primitive_col].values

    x = x - x.mean()
    y = y - y.mean()

    # prevent divide by zero error
    if len(np.unique(x)) == 1:
        return 0

    # consider scipy.stats.linregress for large n cases
    coefficients = np.polyfit(x, y, 1)

    return coefficients[0]



def run_dfs(df,unique_ids,schema_dict,date_col,training_window,groupby_col='enquiryControlNumber'):
    
    try:
        primitive_dict = {

            'mean' : 'mean',
            'sum':'sum',
            'max': 'max',
            'min': 'min',
            'skew' : 'skew',
            'size' : 'size',
            'std' : 'std',
            'count' :'count',
            'last' : 'last',
            'first' :'first',
            'kurtosis' : pd.Series.kurt,
            'trend' : calculate_trend,
            'stdtrend' : calculate_stdtrend,
            'num_unique': pd.Series.nunique,
            'entropy' : entropy_agg,
            'meantrend': calculate_mean_trend}

        
        if (training_window is not None) and (date_col is not None):
            
            df['lower_cutoff_time'] = df['CIBIL_Report_Date'] - pd.DateOffset(months = training_window)

            df = df[df[date_col] >= df['lower_cutoff_time']]
        
        
        features = []

        dummy_dataframe = pd.DataFrame(index=unique_ids)
        
        for where_column_val,where_agg_schema_dict in schema_dict.items():

            where_agg_schema_dict_final = {}
            trend_final_dict = []
            for feature_name,(agg_col,primar_agg) in where_agg_schema_dict.items():
            
                if primar_agg  == 'trend':

                    trend_final_dict.append((feature_name,agg_col,partial(primitive_dict[primar_agg],
                                                                                 date_col=date_col,
                                                                                 primitive_col=agg_col)))

                else:
                    where_agg_schema_dict_final[feature_name] = (agg_col,primitive_dict[primar_agg])

            try:
                where_column, where_value = where_column_val

                # print(where_agg_schema_dict)
                if where_column is not None and where_value is not None:
                    df_subset = df[df[where_column] == where_value]
                else:
                    df_subset = df


                if groupby_col is not None:

                    df_groupby = df_subset.groupby(groupby_col,sort=False)

                    if where_agg_schema_dict_final:
                        agg_feat_dict = df_groupby.agg(**where_agg_schema_dict_final)
                        features.append(agg_feat_dict)
                    

                    for feat_name,agg_col,trend_prim in trend_final_dict:
                        df_trend_subset = df_subset[[groupby_col,agg_col,date_col]]
                        df_trend_groupby = df_trend_subset.groupby(groupby_col,sort=False)
                        dummy_dataframe[feat_name] = df_trend_groupby.agg(trend_prim)[agg_col]
              

            except Exception as e:
                # raise e
                print(f'Error while creating {where_column} where column and {where_value} where value features')
                warnings.warn(f'Error while creating {where_column} where column and {where_value} where value features')
            
        features.append(dummy_dataframe)
        features  = pd.concat(features,axis=1,join='outer',copy=False)
        features = features.astype('float64')
        
        return features
    except Exception as e:
        # raise e
        err_dict_offbook = {'exception':str(e),
                'error':'Error while creating features'}
        print(err_dict_offbook)
        warnings.warn(str(err_dict_offbook))
        return pd.DataFrame(index=unique_ids)
                
        