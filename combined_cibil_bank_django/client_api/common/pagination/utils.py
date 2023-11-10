from collections import OrderedDict


def get_query_params(request):
    order_list = []
    kwargs = OrderedDict()
    query = request.GET
    for key in query.keys():
        if key.endswith('_sort'):
            if query[key] == 'N':
                continue
            elif query[key] == 'D':
                sign = '-'
            elif query[key] == 'A':
                sign = ''
            order_list.append(sign + key[:-len('_sort')])

    return order_list, kwargs, query
