
APP_ID_ERROR = "Application ID not present in the JSON, Please add Application ID in JSON"
ID_ERROR="Identifier not matching"
DATA_ERROR = "data missing. Please add data in JSON"
FEATURE_ERROR = "Could not generate Feature, due to internal server error"
ERROR_CODE_93 = "Unknown internal server error"
ERROR_CODE_94 = "Error in loading data" 
ERROR_CODE_95 = "Missing mandatory data"
ERROR_CODE_97 = ""
ERROR_CODE_98 = "Error in processing data"
ERROR_CODE_99 = "Error in generating features"

def generate_error_dict(err_code, err_type, status_code, err_string, primary_key):
    return {'application_id': primary_key,
           'Error Code': err_code,
           'Error Type': err_type,
           'Status Code': status_code,
           'Error String': err_string}