
import os
import pandas as pd
# from ....json_mapper_new.src.json_data_segregator import JsonDataSegregator
# import json
import multiprocessing as mp
from ..constants import DATASETS





def read_data(path, dataset_name, usecols_):
    if os.path.exists(path):
        print(f'reading {dataset_name} from {path}')
        df_cols = set(pd.read_csv(path, nrows=1).columns)
        usecols_ = list(usecols_.intersection(df_cols))
        return pd.read_csv(path, usecols=usecols_).reset_index(drop=True)

    else:
        print(dataset_name + ' not found at ' + str(path))
        return None



def load_data_from_data_dict(data_path_dict, RENAME_DICTS):
    dfd = {}
    for dataset_name in DATASETS:
        if dataset_name in RENAME_DICTS:
            usecols_ = set(
                [v for k, v in RENAME_DICTS[dataset_name].items() if v != ''])

            if dataset_name in data_path_dict:
            
                if isinstance(data_path_dict[dataset_name], pd.DataFrame):
                    print(f'reading {dataset_name}')
                    df_cols = set(data_path_dict[dataset_name].columns)
                    usecols_ = list(usecols_.intersection(df_cols))
                    dfd[dataset_name] = data_path_dict[dataset_name][usecols_]
                    dfd[dataset_name] = dfd[dataset_name].reset_index(drop=True)
            
#                 elif isinstance(data_path_dict[dataset_name], str):
#                     path = data_path_dict[dataset_name]
#                     dataset = read_data(path, dataset_name, usecols_)
#                     if dataset is not None:
#                         dfd[dataset_name] = dataset
#                     else:
#                         print(dataset_name + ' not found ')
            else:
                print(dataset_name + ' not found ')
        else:
            print(dataset_name + ' not found in RENAME_AND_DTYPE DICT')
    
    return dfd


def load_cibil_json(path):
    with open(path,'rb') as f:
        temp_json = json.load(f)
    
    return temp_json



def load_chunks_of_data(chunk_to_load,n_jobs):
    pool = mp.Pool(n_jobs)
    result = pool.imap_unordered(load_cibil_json,chunk_to_load)
    result = list(result)
    pool.close()
    pool.join()
    return result


# def load_data_from_data_list(data_path_dict, RENAME_DICTS, YAML_PATH, n_jobs):

#     dfd = {}

#     if len(data_path_dict) > 0:

#         if isinstance(data_path_dict[0],str):

#             if n_jobs > 1:
#                 data_path_dict = load_chunks_of_data(data_path_dict,n_jobs)
#             else:
#                 data_path_dict = [load_cibil_json(single_json) for single_json in data_path_dict]
        

#         jds_object = JsonDataSegregator(YAML_PATH=YAML_PATH, OUTPUT_TYPE=1)
#         if len(data_path_dict) > 1:
            
#             data = jds_object.unpack_batch_json(data_path_dict, n_jobs=n_jobs)
#         else:
#             data_dict = jds_object.unpack_json(data_path_dict[0])
#             data = {}
#             for dataset,data_json in data_dict.items():
#                 if dataset!= 'informative_dict':
#                     if isinstance(data_json,list):
#                         data[dataset] = pd.DataFrame(data_json)
#                     else:
#                         data[dataset] = pd.DataFrame([data_dict['informative_dict']])
                  
#         for dataset_name in DATASETS:
#             if dataset_name in RENAME_DICTS:
#                 usecols_ = set(
#                     [v for k, v in RENAME_DICTS[dataset_name].items() if v != ''])
#                 if dataset_name in data:
#                     df_cols = set(data[dataset_name].columns)
#                     usecols_ = list(usecols_.intersection(df_cols))
#                     dfd[dataset_name] = data[dataset_name][usecols_]
#                     dfd[dataset_name] = dfd[dataset_name].reset_index(drop=True)
#                 else:
#                     print(dataset_name + ' not found in JsonDataSegregator Output Dict')
                    
#             else:
#                 print(dataset_name + ' not found in RENAME_AND_DTYPE DICT')
#     else:
#         print('Data List is empty')

#     return dfd
