from rest_framework.permissions import BasePermission
from .models import ApiUser


class ManagerPermissions(BasePermission):
    allowed_user_roles = [ApiUser.MANAGER]

    def has_permission(self, request, view):
        user_group = request.user.groups.filter(
            name__in=self.allowed_user_roles)
        if len(user_group) > 0:
            is_allowed_user = True
        else:
            is_allowed_user = False

        return is_allowed_user


class ScoreApiUserPermissions(BasePermission):
    allowed_user_roles = [ApiUser.SCOREAPIUSER]

    def has_permission(self, request, view):
        user_group = request.user.groups.filter(
            name__in=self.allowed_user_roles)
        if len(user_group) > 0:
            is_allowed_user = True
        else:
            is_allowed_user = False

        return is_allowed_user


class LimitedScoreApiUserPermissions(BasePermission):
    allowed_user_roles = [ApiUser.LIMITEDSCOREAPIUSER]

    def has_permission(self, request, view):
        user_group = request.user.groups.filter(
            name__in=self.allowed_user_roles)
        if len(user_group) > 0:
            is_allowed_user = True
        else:
            is_allowed_user = False

        return is_allowed_user
