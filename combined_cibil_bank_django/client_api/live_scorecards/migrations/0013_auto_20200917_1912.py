# Generated by Django 2.2.3 on 2020-09-17 19:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('live_scorecards', '0012_auto_20200917_1721'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='livedeploymentmodel',
            name='probability_threshold',
        ),
        migrations.AddField(
            model_name='threshold',
            name='probability_threshold',
            field=models.DecimalField(
                decimal_places=10,
                default=0,
                max_digits=12),
            preserve_default=False,
        ),
    ]
