import os
import math
from .serializers import LiveScoreSerializer
from common.pagination.utils import get_query_params
from live_scorecards.models import LiveScore
from .models import LiveDataTransformer, LiveDeploymentPipelineGroup, \
    LiveDeploymentModel
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework import viewsets
#from ml_implementation.ml_modules.ml_explainability import shap_values
from .lib import check_json_types
from .serializers import LiveDataTransformerSerializer
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
#import shap

import logging
logger = logging.getLogger(__name__)

@csrf_exempt
@api_view(['POST'])
@permission_classes((AllowAny,))
def run_pipeline(request, pipeline_name):
    if request.method == 'POST':
        #print('here')
        pipeline = LiveDeploymentPipelineGroup.objects.get(
            pipeline_group_name=pipeline_name)

        type_dict = pipeline.json_type_dict
        if type_dict:
            errors, request_json = check_json_types(
                type_dict, request.data, skip_outer_keys=[
                    "ID", 'account_type', 'generate_shap'])
            if errors:
                return JsonResponse(errors, safe=False, status=400)
        else:
            request_json = request.data

        try:
#             import pdb;
#             pdb.set_trace()
            final_flag, prediction, final_features, transformed_df, \
                percentile, monsoon_score, top_5, bottom_5 = \
                pipeline.run_pipeline(request_json, request_absolute_uri=request.build_absolute_uri())
            if isinstance(final_features, dict):
                # error returned
                error_status = _get_error_status(final_features)
                final_features['error_string'] = _handle_error_str(
                    final_features['error_string']
                )
                final_features['ID'] = request_json['ID']
                logger.error(final_features)
                return JsonResponse(
                    final_features, safe=False, status=error_status)

#             new_score = LiveScore()

#             filename = str(request.user.pk) + '_shap_' + request_json['ID']

#             new_score.user = request.user
#             new_score.approved_flag = final_flag
#             new_score.request_data = request.data
#             new_score.prediction = prediction.tolist()[0]

#             new_score.request_id = request_json['ID']
#             new_score.pipeline = pipeline
#             new_score.monsoon_score = monsoon_score
#             new_score.percentile = percentile
#             new_score.save()
#             import pdb;
#             pdb.set_trace()
            return_dict = final_features.to_dict('r')[0]
            if final_flag is not None:
                return_dict.update({'monsoon_flag': final_flag})
            if percentile is not None:
                return_dict.update({'percentage': percentile})
#             if monsoon_score is not None:
#                 return_dict.update({'monsoon_score': monsoon_score})
#             if top_5:
#                 return_dict.update({'variables_that_reduce_risk': top_5})
#             if bottom_5:
#                 return_dict.update({'variables_that_increase_risk': bottom_5})
            return JsonResponse(return_dict)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            if isinstance(e.args[0], dict):
                error_dict = e.args[0]
                error_status = _get_error_status(error_dict)
                error_dict['error_string'] = _handle_error_str(
                    error_dict['error_string']
                )
                try:
                    error_dict['ID'] = request_json['ID']
                except BaseException:
                    pass
                logger.error(error_dict)
                return JsonResponse(e.args[0],
                                    status=error_status,
                                    safe=False)

            else:
                # unexpected error
                logger.exception(e)
                error_dict = {
                    'error_code': 99,
                    'error_type': 'ServerError',
                    'error_values': str(e)
                }
                try:
                    error_dict['ID'] = request_json['ID']
                except BaseException:
                    pass

                return JsonResponse(error_dict, status=500, safe=False)


def _get_error_status(error_dict):
    if error_dict['error_code'] < 90:
        return 400
    else:
        return 500


def _handle_error_str(error_str):
    if isinstance(error_str, Exception):
        return str(error_str)
    return error_str


@api_view(['POST'])
def run_transformer(request, pipeline_name):
    pass


@api_view(['POST'])
def run_model(request, pipeline_name, model_name):
    if request.method == 'POST':
        pipeline = LiveDeploymentPipelineGroup.objects.get(pk=pk)

        if int(pk) == 1:
            request.data['account_type'] = 'HL'
            model_name = 'vastu_bureau_hl_model_' + model_number
#             pipeline = LiveDeploymentModel.objects.get(ml_model_name=model_number)
        else:
            request.data['account_type'] = 'LP'
            model_name = 'vastu_bureau_pl_model_' + model_number
        model = LiveDeploymentModel.objects.get(ml_model_name=model_name)

        type_dict = pipeline.json_type_dict
        if type_dict:
            errors, request_json = check_json_types(
                type_dict, request.data, skip_outer_keys=[
                    "ID", 'account_type', 'generate_shap'])
        else:
            request_json = request.data
        if errors:
            return JsonResponse(errors, safe=False, status=400)

        try:
            request_json = request.data
#             pipeline_first = LiveDeploymentModelPipeline.objects.get(user=user, pipeline_group = self, is_first=True)
#              prediction, final_features, final_features_df, final_model, monsoon_score, perc = \
#                 pipeline_first.live_model.get_prediction(raw_features_json)

            prediction, final_features, final_features_df, final_model, perc = \
                model.get_prediction(request_json)
            explainers_list = []
            for item in final_model.fitted_models:
                explainers_list.append(shap.TreeExplainer(item))

#             def get_averaged_explainer_value(explainers_list):
# return np.mean([item for item in map(lambda x: x.expected_value,
# explainers_list)])

#             def get_average_shap_val(explainers_list, df_to_explain):
# return np.mean(np.vstack([explainer.shap_values(df_to_explain)[0] for
# explainer in explainers_list]), axis=0)

#             def get_top_fives_col_names(explainers_list, df_to_explain):
#                 # returns best, worst
#                 shap_val = np.mean(np.vstack([explainer.shap_values(df_to_explain.values)[0]
#                      for explainer in explainers_list]), axis=0)
#                 rename_dict = {item: clean_feature_name(item, False) for item in df_to_explain.columns}
#                 df_to_explain.rename(columns=rename_dict, inplace=True)
#                 shap_df = pd.DataFrame({'shap_val': shap_val, 'column':df_to_explain.columns, 'values': df_to_explain.iloc[0].tolist()})
#                 shap_df = shap_df.sort_values('shap_val')
#                 top_5 =  shap_df.iloc[0:5]
#                 bottom_5 = shap_df.iloc[-5:]

#                 return top_5, bottom_5

            top_5, bottom_5 = shap_values.get_top_fives_col_names(
                explainers_list, final_features)
            top_5_features = [
                item if item else 0 for item in top_5['values'].fillna('').tolist()]
            top_5_features_name = top_5['column'].tolist()
            bottom_5_features = [
                item if item else 0 for item in bottom_5['values'].fillna('').tolist()]
            bottom_5_features_name = bottom_5['column'].tolist()
            ID = request.data['ID']
            return JsonResponse({
                'prediction': prediction.tolist()[0],
                #                     'monsoon_flag': model.get_approved_flag(prediction.tolist()[0]),
                #                     'monsoon_score': monsoon_score,
                #                     'percentage': percentile,
                'ID': ID,
                'top_5_variables': top_5_features_name,
                'bottom_5_variables': bottom_5_features_name,
                'top_5_variable_values': top_5_features,
                'bottom_5_variable_values': bottom_5_features,
            })
        except KeyError as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse([{
                'error': str(e),
                'errorType': 'KeyError',
            }], status=400, safe=False)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse([{
                'error': 'Internal Server Error',
                'errorType': 'ServerError',
                'errorString': str(e)
            }], status=400, safe=False)


class TransformerViewset(viewsets.ModelViewSet):
    serializer_class = LiveDataTransformerSerializer

    def get_queryset(self):
        return LiveDataTransformer.objects.filter(user=self.request.user)


@api_view(['GET'])
def get_reasons():
    pass


@api_view(['POST'])
def get_transformed_data(request, pipeline_name, model_name):
    pass


@api_view(['GET'])
def past_scores(request):
    if request.method == 'GET':
        current_page = int(request.GET.get('current_page'))
        items_per_page = int(request.GET.get('items_per_page'))
        results_count = LiveScore.objects.count()
        start_index = (current_page - 1) * items_per_page
        end_index = (current_page - 1) * items_per_page + items_per_page

        pages = math.ceil(results_count / items_per_page)
        order_list, kwargs, query = get_query_params(request)

        results = LiveScore.objects.filter(
            user=request.user).order_by(
            *
            order_list)[
            start_index: end_index]
        past_preds = LiveScore.objects.filter(
            user=request.user).order_by('prediction').values_list(
            'prediction', flat=True)

        response = []

        for item in results:
            chart_page = {}
            if item.approved_flag:
                fa_class = 'fas fa-check'
            else:
                fa_class = 'fas fa-times text-danger'

            shap = item.shap
            top_5_details = []
            bottom_5_details = []

            for top_5_val, top_5_key in zip(
                    shap.top_5_features, shap.top_5_features_name):
                top_5_details.append({
                    'key': top_5_key,
                    'value': top_5_val
                })
            for bottom_5_val, bottom_5_key in zip(
                    shap.bottom_5_features, shap.bottom_5_features_name):
                bottom_5_details.append({
                    'key': bottom_5_key,
                    'value': bottom_5_val
                })

            card_details = []
            card_details.append({
                'key': 'ID',
                'value': item.request_id
            })

            cards = [
                {
                    'title': 'Applicant Details',
                    "css_class": "col-lg-4 col-md-12",
                    "details": card_details,
                },
                {
                    'title': 'Percentile in Applicant Pool',
                    "css_class": 'col-lg-8 col-m-1 col-md-12',
                    'large_text': '{} %'.format(round(item.percentile, 2))
                },
                #  {
                #      'title': 'CIBIL Score',
                #      "css_class": 'col-lg-4 col-md-12',
                #      "border_class": 'border-top-custom',
                #      'gauge_chart': GaugeChart(300, needle_value, "",
                #                                "", cibil_score,
                #                                {
                #                                    "hasNeedle": True,
                #                                    "needleColor": "#333333",
                #                                    "arcColors": ["#003f5c", "#ffd600", "#D50000"][::-1],
                #                                    "arcDelimiters": [33, 66],
                #                                    "rangeLabel": ['300', '900'],

                #                                }
                #                               ).to_dict()
                #  },
                {
                    'title': 'Probability of default(%)',
                    "css_class": 'col-lg-4 col-md-12',
                    "border_class": 'border-top-custom',
                    'gauge_chart': GaugeChart(300, round(item.prediction, 2), "",
                                              "", round(item.prediction, 2),
                                              {
                                                  "hasNeedle": True,
                                                  "needleColor": "#333333",
                                                  "arcColors": ["#003f5c", "#ffd600", "#D50000"],
                                                  "arcDelimiters": [33, 66],
                                                  "rangeLabel": ['0', '100'],

                    }
                    ).to_dict()
                },
                {
                    'title': 'Monsoon Score',
                    "css_class": 'col-lg-4 col-md-12',
                    "border_class": 'border-top-custom',
                    'gauge_chart': GaugeChart(300, round(item.monsoon_score, 2), "",
                                              "", round(item.monsoon_score, 2),
                                              {
                                                  "hasNeedle": True,
                                                  "needleColor": "#333333",
                                                  "arcColors": ["#003f5c", "#ffd600", "#D50000"],
                                                  "arcDelimiters": [33, 66],
                                                  "rangeLabel": ['300', '900'],

                    }
                    ).to_dict()
                },
                {
                    'title': 'Approve/Reject',
                    "css_class": 'col-lg-4 col-md-12',
                    "border_class": 'border-top-custom',
                    'fa_name': fa_class,
                },
                # {
                #    'title': 'SHAP Analysis for the Individual',
                #    "css_class": 'col-lg-12 col-md-12',
                #    'img_link': shap.stored_image.storage_url,#"https://i.imgur.com/UU6dug9.png",
                #    'img_fluid': True,
                # },
                {
                    'title': "Shap Top 5 features",
                    'css_class': 'col-lg-6 col-md-12',
                    'details': top_5_details
                },
                {
                    'title': "Shap Bottom 5 features",
                    'css_class': 'col-lg-6 col-md-12',
                    'details': bottom_5_details
                },

            ]
            chart_page['cards'] = cards
            response.append({
                'score_data': LiveScoreSerializer(item).data,
                'chart_page': chart_page
            })

        return JsonResponse({
            'total_pages': pages,
            'result': response,
            'headers': ['ID', 'Request Id', 'Prediction', 'Approved Flag', 'Created At'],
            'column_names': ['id', 'request_id', 'prediction', 'approved_flag', 'created_at']
        }, safe=False)

    
@api_view(['GET'])
@permission_classes((AllowAny,))
def get_model_info(request):
    pipeline_group_name = LiveDeploymentPipelineGroup.objects.first().pipeline_group_name
    model_names = list(LiveDeploymentModel.objects.all().values_list("ml_model_name",flat=True))
    return JsonResponse({
        "pipeline_name":pipeline_group_name,
        "models":model_names
    })