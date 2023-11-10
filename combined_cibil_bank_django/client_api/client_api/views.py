from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_200_OK
)
from rest_framework.response import Response
from pathlib import Path


@csrf_exempt
@api_view(['POST'])
@permission_classes((AllowAny,))
def login(request):
    username = request.data.get('username')
    password = request.data.get('password')
    if username is None or password is None:
        return Response({'error': 'Please provide both username and password'},
                        status=HTTP_400_BAD_REQUEST)
    user = authenticate(username=username, password=password)

    if not user:
        return Response({'error': 'Invalid Credentials'},
                        status=HTTP_404_NOT_FOUND)
    token, _ = Token.objects.get_or_create(user=user)
    return Response({'token': token.key, 'username': user.username},
                    status=HTTP_200_OK)


@api_view(['GET'])
def get_access_log(request):
    try:
        reqlog = _get_log_file('model_req')
        return JsonResponse({'access_log': reqlog})
    except BaseException:
        return JsonResponse({'error': "unable to generate access logs"})


@api_view(['GET'])
def get_error_log(request):
    try:
        errlog = _get_log_file('model_error')
        return JsonResponse({'error_log': errlog})
    except BaseException:
        return JsonResponse({'error': "unable to generate error logs"})


def _get_log_file(log_file_name):
    log_text = ''
    BASE_LOG_DIR = os.path.join(BASE_DIR, 'log')
    log_files = sorted(Path(BASE_LOG_DIR).iterdir(), key=os.path.getmtime)
    for file in log_files:
        if log_file_name in file:
            with open(os.path.join(BASE_LOG_DIR, file)) as fp:
                log_text += fp.readlines()
    return log_text
