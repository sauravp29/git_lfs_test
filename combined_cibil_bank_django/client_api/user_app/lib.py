from .models import ApiUser


def cmd_user_select():
    for user in ApiUser.objects.all():
        print('PK {} Clientname/Username {}'.format(user.pk, user.username))

    client_pk = int(eval(input('Enter the pk of client: ')))
    client = ApiUser.objects.get(pk=client_pk)
    return client
