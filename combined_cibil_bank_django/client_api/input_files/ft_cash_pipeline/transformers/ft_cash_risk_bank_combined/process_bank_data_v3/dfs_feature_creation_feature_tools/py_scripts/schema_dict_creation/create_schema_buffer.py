import re
import pandas as pd
import numpy as np
import os
import sys
import pickle
import time


def get_index(s, sub_s, ins, from_start=True):
    '''
        THis function will return the index of given substring from a string
        Feature to look from start or end is there to define lookup position.
        Also ins is the position of that instance that needs to be returned
        Args: s: string
          sub_s: substring
          from_start: position to start
    '''
    all_instances = [i for i in range(len(s)) if s.startswith(sub_s, i)]
    ind = all_instances[ins - 1] if from_start else all_instances[-ins]
    return ind


def get_cond(s, composition_level):
    '''
        This function will output the filter condition that needs to be applied if where clause is present other wise it will return
        s - column_name
        composition_level: is stagewise level stage 1 is composition 1 and stage 2 will be 2
    '', ''
    '''
    try:
        if composition_level != 1:
            return '', ''
        li_cond_col = get_index(s, 'WHERE', composition_level, False) + 6
        ri_cond_col = get_index(s, ' = ', composition_level, False)
        li_cond_val = ri_cond_col + 3
        ri_cond_val = get_index(s, ')', composition_level, False)
        condition_col = s[li_cond_col:ri_cond_col]
        condition_val = s[li_cond_val:ri_cond_val]
    except BaseException:
        condition_col = ''
        condition_val = ''
    return condition_col, condition_val


def get_feat_colname(s, composition_level=1):
    '''
      This function will return the columns name that needs to aggregated upon
      There will be two composition level
      comp - 1 Denotes stage 1 column names
      comp - 2 Denotes column names are stage 2 column_names needs for the stage 3 data
    '''
    try:
        li = get_index(s, 'transactions', 1) + 13
    except BaseException:
        return ''
    try:
        ri = get_index(s, ' WHERE', composition_level, True)
    except BaseException:
        ri = get_index(s, ')', composition_level,
                       False) + composition_level - 1
    colname = s[li:ri]
    # Why?
    if colname.endswith('date'):
        colname = colname[:-6]
    return colname


def get_agg(s):
    '''
        Return simple aggregation function name
    '''
    li = 0
    ri = s.index('(')
    agg = s[li:ri]
    return agg


def get_path(s, composition_level):
    '''
        Return the stage_name
    '''
    li = get_index(s, ')_', 1, False) + 1
    path = s[li:]
    return path


def create_schema_buffer(s, schema_dict, win_num=None, feat_num=None):
    '''
        s - column name
        schema_dict- empty_dict/iteratively updated dict after each operation
        win_num - for internal use (Recursion Base Conditions)
        feat_num - for internal use (Recursion Base Conditions)
    '''
    composition_level = s.count(')')
    if not composition_level:
        return schema_dict
    if (win_num is None) and (feat_num is None):
        path = get_path(s, composition_level)
    else:
        # change this behavior to compute only the number of days needed for
        # each intermediate columns
        feat_num = 300
        path = f'_intermediate_{win_num}_features_{feat_num}'
    # banaid to handle unintended trnaslations that were built by DFS- must be
    # removed one day
    if (path == '_features_300') and (composition_level == 2):
        return schema_dict

    if win_num is None:
        pat_win = re.compile('_window_[0-9]*')
        win_num = pat_win.findall(path)
        win_num = int(win_num[0][8:]) if len(win_num) else None
    if feat_num is None:
        pat_feat = re.compile('_features_[0-9]*')
        feat_num = pat_feat.findall(path)
        feat_num = int(feat_num[0][10:]) if len(feat_num) else None
    schema_dict[(path, win_num, feat_num, composition_level)] = schema_dict.get(
        (path, win_num, feat_num, composition_level), {})
    uc = schema_dict[(path, win_num, feat_num, composition_level)]
    condition = get_cond(s, composition_level)
    agg = get_agg(s)
    feat_col = get_feat_colname(s, composition_level)
    uc[condition] = uc.get(condition, {})
    uc[condition][feat_col] = list(
        set(uc[condition].get(feat_col, []) + [(agg, s)]))

    if win_num:
        create_schema_buffer(feat_col, schema_dict, win_num, feat_num)
    return schema_dict
