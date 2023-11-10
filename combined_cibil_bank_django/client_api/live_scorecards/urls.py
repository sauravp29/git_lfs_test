from django.conf.urls import include, url
from django.contrib import admin
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register(
    r'transformers',
    views.TransformerViewset,
    basename='LiveTransformer')

app_name = 'live_scorecards'

urlpatterns = [url(r'get_model_info/',
                   views.get_model_info,
                   name='get_model_info'),
               url(r'predict/live/(?P<pipeline_name>\w+)/',
                   views.run_pipeline,
                   name='pipeline'),
               url(r'predict/live/model_two/(?P<pipeline_name>\w+)/(?P<model_name>\w+)',
                   views.run_model,
                   name='run_model'),
               url(r'past-scores/',
                   views.past_scores,
                   name='past_scores')
               ]

