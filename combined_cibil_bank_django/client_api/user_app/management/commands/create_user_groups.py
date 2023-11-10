from django.core.management.base import BaseCommand
from user_app.models import ApiUser, USERGROUPS
from django.contrib.auth.models import Group, Permission

import uuid


class Command(BaseCommand):
    """
        This command creates user groups and permissions for auth purposes.
    """

    def add_arguments(self, parser):
        parser.add_argument(
            'manager_username',
            type=str,
            help='Username for manager user')
        parser.add_argument(
            'scoreapi_username',
            type=str,
            help='Username for manager user')

    def handle(self, *args, **kwargs):
        """
            There are three user groups:
            1. Users who can view all the features of the app i.e managers
            2. Users who can only view the live scorecard page with all the content i.e scoreapiuser
            3. Users who can only view the live scorecard with limited content i.e limitedscoreapiuser
        """
        manager_username = kwargs['manager_username']
        manager_username = kwargs['manager_password']
        manager_email = kwargs['manager_email']
        scoreapi_username = kwargs['scoreapi_username']
        scoreapi_password = kwargs['scoreapi_password']
        scoreapi_email = kwargs['scoreapi_email']
        for group in USERGROUPS:
            new_group, created = Group.objects.get_or_create(name=group)

        # create 2 users:
            # 1. Manager: Add 1st and 2nd group to this user.
            # 2. ApiUser: With only 3rd permission for limited api access

        manager = ApiUser(
            username=manager_username,
            password=manager_password,
            email=manager_email
        )

        scoreapi_user = ApiUser(
            username=scoreapi_username,
            password=scoreapi_password,
            email=scoreapi_email
        )
