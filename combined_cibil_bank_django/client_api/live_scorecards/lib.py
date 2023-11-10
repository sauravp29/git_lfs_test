from .models import LiveDataTransformer, LiveDeploymentModel, \
    LiveDeploymentModelPipeline, LiveDeploymentPipelineGroup
from common.errors import ResponseListError
import json
from datetime import datetime


def check_json_types(data_types, request_json, skip_outer_keys=None):
    """
        data_types: json_type_dict obtained from the pipeline group
    """
    errors = []
    if not skip_outer_keys:
        skip_outer_keys = []
    for outer_key in data_types.keys():
        if outer_key in skip_outer_keys:
            continue
        if outer_key not in request_json:
            errors.append({
                'error': 'The data for the field {} is missing.'.format(outer_key),
                'type': 'MissingDataError',
                'outerKey': outer_key,
            })
        else:
            val = request_json[outer_key]
            if val is None or val == '':
                continue
            elif isinstance(val, list):
                for (index, item) in enumerate(val):
                    for inner_key in data_types[outer_key]:
                        if item is None or item is '' or not isinstance(
                                item, dict):
                            errors.append(
                                {
                                    'error': 'The data for the field {}[{}] is missing'.format(
                                        outer_key,
                                        inner_key),
                                    'type': 'MissingDataError',
                                    'outerKey': outer_key,
                                    'innerKey': inner_key})
                            continue

                        elif inner_key not in item.keys():
                            errors.append(
                                {
                                    'error': 'The data for the field {}[{}] is missing'.format(
                                        outer_key,
                                        inner_key),
                                    'type': 'MissingDataError',
                                    'outerKey': outer_key,
                                    'innerKey': inner_key})
                            continue

                        else:
                            if item[inner_key] is None or item[inner_key] == "":
                                request_json[outer_key][index][inner_key] = item[inner_key]
#                                 request_json[outer_key][index][inner_key] = None
                                continue
                            res, error = check_type(
                                item[inner_key], data_types[outer_key][inner_key], outer_key, inner_key)
                            if res is not None and res != '':
                                request_json[outer_key][index][inner_key] = res
                            if error:
                                errors.append(error)
                if len(val) == 0:
                    errors.append({'error': 'The data for the field {} is missing. Please provide all the inner keys as null even if it has no data.'.format(
                        outer_key), 'type': 'MissingDataError', 'outerKey': outer_key, })

            elif isinstance(val, dict):
                for _key, _val in val.items():
                    if _val is None or _val == '':
                        #                         request_json[outer_key][_key] = None
                        request_json[outer_key][_key] = _val
                        continue
                    res, error = check_type(
                        _val, data_types[outer_key][_key], outer_key, _key)
                    if res is not None and res != "":
                        request_json[outer_key][_key] = res
                    if error:
                        errors.append(error)
            elif isinstance(val, data_types[outer_key]):
                continue
            else:
                errors.append({'error': 'The type for the field {} is wrong. Please pass type {}'.format(
                    outer_key, data_types['outer_key']), 'type': 'MissingDataError', 'outerKey': outer_key, })

    return errors, request_json


def check_type(val, _type, outer_key, inner_key=''):
    try:
        if isinstance(_type, str):
            type_str = 'Date' + ' with format: ' + _type
            datetime.strptime(val, _type)
            res = None
        else:
            type_str = _type.__name__
            res = _type(val)

        return res, None

    except (ValueError, TypeError):
        return None, {'error': 'The data for the field {}[{}] is {}. The value recieved was {}'.format(
            outer_key, inner_key, type_str, val), 'type': 'InvalidDataError', 'innerKey': inner_key, 'outerKey': outer_key}
