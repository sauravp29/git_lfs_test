# This modules defines several classes used in pagination.
import pandas as pd


class DataCell(object):

    def __init__(
            self,
            value,
            client_page_link=False,
            client_page_url=None,
            get_options=None):
        self.value = value
        # its boolean whether there is a link or not
        self.client_page_link = client_page_link
        # actualu link to fetch data from an api that returns chart_page.
        self.client_page_url = client_page_url
        self.get_options = get_options

    def to_dict(self):
        return {
            'value': self.value,
            'client_page_link': self.client_page_link,
            'client_page_url': self.client_page_url,
            'get_options': self.get_options
        }


class HeaderCell(object):

    def __init__(self, column_name, field_name, client_page_link=None):
        self.column_name = column_name
        self.field_name = field_name
        self.client_page_link = client_page_link

    def to_dict(self):
        return {
            'column_name': self.column_name,
            'field_name': self.field_name,
            'client_page_link': self.client_page_link
        }


class HeaderRow(object):

    def __init__(self, header_cells):
        self.row = header_cells

    def to_dict(self):
        result = [item.to_dict() for item in self.row]
        return {
            'row': result
        }


class DataRow(object):

    def __init__(self, data_cells):
        self.row = data_cells

    def to_dict(self):
        result = [item.to_dict() for item in self.row]
        return {
            'row': result
        }


class Table(object):
    """
        Represents a paginated table with data source set as a url.
        Rendered by angular-material-table component of client-ui app.
    """

    def __init__(
            self,
            source_data_url,
            current_page,
            page_size,
            total_rows,
            rows,
            header,
            table_class,
            title=''):
        """
            Attributes
            ----------
            source_data_url: string
                Url of server to get this table object
            current_page: number
                Current Page to be rendered/shown
            page_size: number
                No of rows/page
            total_rows: number
                Total rows of data that have to be paginated
            rows: DataRow[]
                Rows of data for current page. Length is same as page_size
            header: HeaderRow
                Headers to show in table
            table_class: string
                css classes to apply to table while rendering it
            title: string
                Title of the table if any
        """
        self.source_data_url = source_data_url
        self.current_page = current_page
        self.page_size = page_size
        self.total_rows = total_rows
        self.rows = rows
        self.header = header
        self.table_class = table_class
        self.title = title

    def to_dict(self):
        rows_dict = [item.to_dict() for item in self.rows]
        return {
            'source_data_url': self.source_data_url,
            'current_page': self.current_page,
            'page_size': self.page_size,
            'total_rows': self.total_rows,
            'rows': rows_dict,
            'header': self.header.to_dict(),
            'table_class': self.table_class,
            'title': self.title
        }

    @staticmethod
    def from_dataframe(
            df,
            source_data_url,
            current_page,
            page_size,
            header,
            table_class,
            title='',
            client_page_url='',
            client_page_get_options_getter=None,
            get_options_getter_kwargs=None):
        """
            client_page_options_getter: callable function which return a list of tuples() of length 2 pairs for get request for action specified in action column.
                                        First item in tuple is key of get option and second item is it's value.
        """
        start_index, end_index = get_indices_paginate(current_page, page_size)
        rows = df.iloc[start_index: end_index, :]
        rows.fillna('', inplace=True)
        rows_table = []
        for index, item in rows.iterrows():
            row = []
            for col in header.row:
                if not col.client_page_link:
                    val = item[col.field_name]
                    if pd.isna(val):
                        val = ''
                    row.append(DataCell(val))
                else:
                    if not get_options_getter_kwargs:
                        get_options_getter_kwargs = {}
                    if client_page_get_options_getter:
                        row.append(
                            DataCell(
                                'Details',
                                True,
                                client_page_url,
                                client_page_get_options_getter(
                                    item,
                                    **get_options_getter_kwargs)))
                    else:
                        row.append(DataCell('Details', True, client_page_url))
            rows_table.append(HeaderRow(row))
        table = Table(
            source_data_url,
            current_page,
            page_size,
            df.shape[0],
            rows_table,
            header,
            table_class)
        return table


def get_indices_paginate(current_page, page_size):
    start_index = (current_page - 1) * page_size
    end_index = start_index + page_size
    return start_index, end_index
