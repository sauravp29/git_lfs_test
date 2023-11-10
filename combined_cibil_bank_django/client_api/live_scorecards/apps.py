from django.apps import AppConfig
from django.core.cache import caches

import logging
logger = logging.getLogger(__name__)


class LiveScorecardsConfig(AppConfig):
    name = 'live_scorecards'

    def ready(self):
        from .models import LiveDeploymentModel
        #from ml_implementation.ml_modules.ml_explainability import shap_values

#         # caching all models
#         for ml_model in LiveDeploymentModel.objects.all():
#             # this caches the model
#             logger.error('caching model')
#             trained_model = ml_model.get_default_model()

#             # caching shap objects
#             if ml_model.explainability_option == 'shap':
#                 lcache = caches['localmem']
#                 logger.info('caching shap')

#                 shap_cache_key = ml_model.user.username + \
#                     '_' + ml_model.ml_model_name + '_shap'
#                 explainers_list = shap_values.get_explainers_list(
#                     trained_model)
#                 lcache.set(
#                     shap_cache_key,
#                     explainers_list,
#                     None)  # save in the cache
