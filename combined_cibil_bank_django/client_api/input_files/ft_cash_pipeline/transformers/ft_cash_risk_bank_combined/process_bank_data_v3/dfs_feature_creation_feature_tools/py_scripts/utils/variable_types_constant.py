import featuretools
# TODO: Might Need to fix the column mapping once the final column names
# are being finalised as per the feautre selection.
STAGE_1_VARIABLE_TYPE = {
    'date': featuretools.variable_types.variable.DatetimeTimeIndex,
    'datediff_txn': featuretools.variable_types.variable.Numeric,
    'APP_ID': featuretools.variable_types.variable.Id,
    'amount': featuretools.variable_types.variable.Numeric,
    'balance': featuretools.variable_types.variable.Numeric,
    'type': featuretools.variable_types.variable.Categorical,
    'ind': featuretools.variable_types.variable.Categorical,
    'txn_classified_as': featuretools.variable_types.variable.Categorical,
    'nach_type': featuretools.variable_types.variable.Categorical,
    'day_type-type': featuretools.variable_types.variable.Categorical,
    'nach_type-type': featuretools.variable_types.variable.Categorical,
    'is_national_holiday-type': featuretools.variable_types.variable.Categorical,
    'is_national_holiday-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'nach_type-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'day_type-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'is_redundant-nach_type': featuretools.variable_types.variable.Categorical,
    'is_redundant-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'first_7_days-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'last_7_days-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'amount_0-1000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'amount_1000-5000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'amount_5000-10000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'balance_l-0-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'balance_0-10000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'balance_10000-20000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'balance_20000-50000-txn_classified_as': featuretools.variable_types.variable.Categorical}

STAGE_2_VARIABLE_TYPE = {
    'date': featuretools.variable_types.variable.DatetimeTimeIndex,
    'datediff_txn': featuretools.variable_types.variable.Numeric,
    'APP_ID': featuretools.variable_types.variable.Id,
    'amount': featuretools.variable_types.variable.Numeric,
    'balance': featuretools.variable_types.variable.Numeric,
    'type': featuretools.variable_types.variable.Categorical,
    'ind': featuretools.variable_types.variable.Categorical,
    'txn_classified_as': featuretools.variable_types.variable.Categorical,
    'nach_type': featuretools.variable_types.variable.Categorical,
    'day_type-type': featuretools.variable_types.variable.Categorical,
    'nach_type-type': featuretools.variable_types.variable.Categorical,
    'is_national_holiday-type': featuretools.variable_types.variable.Categorical,
    'is_national_holiday-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'nach_type-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'day_type-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'is_redundant-nach_type': featuretools.variable_types.variable.Categorical,
    'is_redundant-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'first_7_days-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'last_7_days-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'amount_0-1000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'amount_1000-5000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'amount_5000-10000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'balance_l-0-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'balance_0-10000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'balance_10000-20000-txn_classified_as': featuretools.variable_types.variable.Categorical,
    'balance_20000-50000-txn_classified_as': featuretools.variable_types.variable.Categorical}


STAGE_3_VARIABLE_TYPE = {
    'date': featuretools.variable_types.variable.DatetimeTimeIndex,
    'APP_ID': featuretools.variable_types.variable.Id}

VARIABLE_MAPPING_DICT = {'stage_1': STAGE_1_VARIABLE_TYPE,
                         'stage_2': STAGE_2_VARIABLE_TYPE,
                         'stage_3': STAGE_3_VARIABLE_TYPE}
