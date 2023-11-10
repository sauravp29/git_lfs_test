# Generated by Django 2.2.3 on 2020-09-21 20:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('live_scorecards', '0013_auto_20200917_1912'),
    ]

    operations = [
        migrations.AlterField(
            model_name='livescore',
            name='approved_flag',
            field=models.CharField(
                max_length=255,
                null=True),
        ),
        migrations.AlterField(
            model_name='threshold',
            name='action_if_greater_prob',
            field=models.CharField(
                choices=[
                    ('model',
                     'Model'),
                    ('true',
                     True),
                    ('false',
                     False),
                    ('threshold',
                     'threshold'),
                    ('string',
                     'string'),
                    ('break',
                     None)],
                max_length=255),
        ),
        migrations.AlterField(
            model_name='threshold',
            name='action_if_lower_prob',
            field=models.CharField(
                choices=[
                    ('model',
                     'Model'),
                    ('true',
                     True),
                    ('false',
                     False),
                    ('threshold',
                     'threshold'),
                    ('string',
                     'string'),
                    ('break',
                     None)],
                max_length=255),
        ),
    ]
