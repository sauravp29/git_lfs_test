from django.http import JsonResponse

from common.errors import ResponseListError


class ExceptionMiddleware:

    def __init__(self, get_response):
        """Initialise the Exception Middleware"""

        self.get_response = get_response

    def __call__(self, request):
        """Default call for every request passing through the middleware layer"""

        response = self.get_response(request)
        return response

    @staticmethod
    def process_exception(request, exc):
        """ Handles exceptions raised by the view layer

           :param request: The incoming request.
           :type request: `django.http.request.HttpRequest`

           :param exc: the exception raise from anywhere in the view or service layer
           :type exc: subcalss of the Exception class

           :returns: response: based upon the exception received in the exception parameter
                           None: if the raised exception is not being handled,
                                   so that Django's default exception handling can kick-in
        """
        # user errors
        if isinstance(exc, ResponseListError):
            response = JsonResponse({
                'errors': exc.errors
            }, status=400)
        else:
            # Framework level exceptions
            response = JsonResponse({
                'errors': ['Server Error 500']
            }, status=500)

        return response
