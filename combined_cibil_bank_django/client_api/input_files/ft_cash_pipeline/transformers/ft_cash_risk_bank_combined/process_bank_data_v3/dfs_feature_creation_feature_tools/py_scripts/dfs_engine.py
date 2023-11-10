from custom_exceptions import *
import pickle
from utils.run_dfs_pipeline import *
from multiprocessing import cpu_count


class SmsDFSEngine():
    def __init__(self, SCHEMA_DICT_PATH_OR_DICT):

        self.SCHEMA_DICT = self.load_schema_dict(
            (SCHEMA_DICT_PATH_OR_DICT)

    def load_schema_dict(self, DICT_OR_PATH):
        try:
            if isinstance(DICT_OR_PATH, str):
                return pickle.load(open(PATH, 'rb'))
            elif isinstance(DICT_OR_PATH, dict):
                return DICT_OR_PATH
            else:
                raise Exception("Type not supported for the given Schema Dict")
        except Exception as e:
            raise SchemaDictLoadError(e)
