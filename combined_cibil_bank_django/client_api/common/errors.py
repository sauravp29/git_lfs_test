class ResponseListError(Exception):
    """
        Class to manage errors recieved from various views.
        It accepts a message and errors parameter where
        errors is a list of string errors
    """

    def __init__(self, message, errors):
        super(ResponseListError, self).__init__(message)

        self.errors = errors
