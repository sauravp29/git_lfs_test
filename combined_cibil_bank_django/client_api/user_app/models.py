# -*- coding: utf-8 -*-


from django.db import models
from django.contrib.auth.models import AbstractUser

USERGROUPS = [
    'manager',
    'scoreapiuser',
    'limitedscoreapiuser'
]


class ApiUser(AbstractUser):
    MANAGER = 'manager'
    SCOREAPIUSER = 'scoreapiuser'
    LIMITEDSCOREAPIUSER = 'limitedscoreapiuser'
