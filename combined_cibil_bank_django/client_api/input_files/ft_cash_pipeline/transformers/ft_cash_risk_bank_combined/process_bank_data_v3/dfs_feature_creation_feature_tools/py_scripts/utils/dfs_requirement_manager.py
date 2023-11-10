from featuretools import EntitySet, make_temporal_cutoffs
from pandas import DataFrame, merge, Timedelta
from .variable_types_constant import VARIABLE_MAPPING_DICT


def intialise_variable_types(STAGE_NAME):
    variable_type_mapping = VARIABLE_MAPPING_DICT[STAGE_NAME]
    return variable_type_mapping


def intialise_dfs(batch_dataframe, STAGE_NAME, ID='txn_base'):
    es = EntitySet(id=ID)
    variable_type_mapping = intialise_variable_types(STAGE_NAME)
    variable_type_mapping_filtered = {
        k: v for k,
        v in variable_type_mapping.items() if k in batch_dataframe.columns.tolist()}
    index_flag = True
    if 'txn_index' in batch_dataframe.columns:
        index_flag = False

    es = es.entity_from_dataframe(
        'transactions',
        dataframe=batch_dataframe,
        make_index=index_flag,
        index='txn_index',
        variable_types=variable_type_mapping_filtered,
        time_index='date',
        already_sorted=True)
    es = es.normalize_entity('transactions', 'ids', 'APP_ID')
    es.add_last_time_indexes()
    return es


def create_stage_2_requirements(entity_set, batch_created_date_mapping):

    first_txn_time = entity_set['ids'].df.first_transactions_time.to_dict()
    f_t_t_df = DataFrame(first_txn_time.items(), columns=[
        'instance_id', 'first_txn_time'])
    es_ids_to_first_txn_time = merge(
        batch_created_date_mapping, f_t_t_df, on='instance_id', how='inner')
    dfa_instance_ids = es_ids_to_first_txn_time['instance_id'].copy(
        deep=True).values
    dfa_instance_time = es_ids_to_first_txn_time['time'].copy(deep=True).values
    es_first_txn_time = es_ids_to_first_txn_time['first_txn_time'].copy(
        deep=True).values
    del first_txn_time, f_t_t_df, es_ids_to_first_txn_time
    return dfa_instance_ids, dfa_instance_time, es_first_txn_time


def calculate_temporal_cutoffs(
        dfa_instance_ids,
        dfa_instance_time,
        es_first_txn_time,
        cutoff_time):

    # Hard Coded for the 1 day and 1 month needs modification if in future
    # requirements changes.
    cutoff_string = str(cutoff_time) + ' D'
    temporal_cutoff = make_temporal_cutoffs(dfa_instance_ids,
                                            dfa_instance_time,
                                            window_size=Timedelta(
                                                cutoff_string),
                                            start=es_first_txn_time)
    return temporal_cutoff
