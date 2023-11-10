import re
from collections import OrderedDict


def get_required_columns(list_of_features):
    cond_dict = OrderedDict()
    indi_lis = OrderedDict()
    for s in list_of_features:
        # cond.append(re.search(r'WHERE (.*?) =', s).group(1))
        if 'WHERE' in s:
            cond_dict.update({s[s.find('WHERE') + 6: s.find('=') - 1]: ''})
            for i in cond_dict:
                if '-' in i:
                    first_name = i[:i.rfind('-')]
                    second_name = i[i.rfind('-') + 1:]
                    if second_name.isdigit() or second_name == '':
                        continue
                    else:
                        indi_lis.update({first_name: '', second_name: ''})
    indi_lis.update(cond_dict)
    return list(indi_lis.keys())
