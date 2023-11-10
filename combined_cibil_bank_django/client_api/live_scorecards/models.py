import io
import os
import sys
import imp
import uuid
import json
import pickle
import urllib
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.base import TransformerMixin

from django.db import models
from django.core.cache import caches
from django.templatetags.static import static
from picklefield.fields import PickledObjectField

from user_app.models import ApiUser
from client_api.settings import BASE_DIR

# from ml_implementation.ml_modules.ml_auto.custom_classifier import CustomClassifier
# from ml_implementation.ml_modules.ml_auto import encoding
# from ml_implementation.ml_modules.ml_explainability import shap_values
# from ml_implementation.ml_modules.ml_explainability.surrogate_model_explainability import generating_most_imp_features


class NominalCalibration(TransformerMixin):
    def __init__(self, no_of_delinquents, preds):
        self.no_of_delinquents = no_of_delinquents
        self.preds = preds

    def fit(self, X, y=None):
        self.nominal_weight = self.no_of_delinquents / sum(self.preds)
        return self

    def transform(self, X):
        X = X * self.nominal_weight
        return X

    def get_params(self, **kwargs):
        return {
            'no_of_delinquents': self.no_of_delinquents,
            'preds': self.preds,
        }

    def save_model(self, file_name):
        if file_name:
            with open(file_name, "wb") as out_file:
                pickle.dump({"params": self.get_params(),
                             'nominal_weight': self.nominal_weight}, out_file)
                return file_name

    @staticmethod
    def load_model(file_name):
        """
        Loads a model from saved picke of fitted models and Estimator params.
        return an Estimator instance
        """
        _dict = pickle.load(open(file_name, "rb"))
        est_ = NominalCalibration(**_dict['params'])
        est_.nominal_weight = _dict['nominal_weight']
        return est_

    @staticmethod
    def load_model_bytes(_bytes):
        """
        Loads a model from saved picke of fitted models and Estimator params.
        return an Estimator instance
        """
        _dict = pickle.loads(_bytes)
        est_ = NominalCalibration(**_dict['params'])
        est_.nominal_weight = _dict['nominal_weight']
        return est_


def _get_uuid():
    return uuid.uuid4().hex


class LiveDataTransformer(models.Model):
    """
        Saves the tranformer code for transforming json data to required features
        Attributes:
        -----------

        pickled_transformer:
        -------------------
        A binary python file with the code for the transformer.
        The code should have have  a transform function which
        takes a single argument as the dataframe/dict to be transformed to final features output.

    """
    DYNAMIC_MOD_PATH = os.path.join(BASE_DIR, '_dynamic_modules')
    user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    transformer_name = models.CharField(max_length=255)
    # created by default for each transformer
    folder_uuid = models.CharField(max_length=255, default=_get_uuid)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    MODULE_NAME = 'custom_transformer'

    class Meta:
        unique_together = ('user', 'folder_uuid')

    @staticmethod
    def create(transformer_folder, transformer_name, folder_uuid, user):
        """
            Attributes:
                transformer_content bytes:
                    binary data for json to raw features transformer such as from file io read() functiom etc.
                    transform(json)
                name str: this name will be used as the name of dynamic module created to use this transformer

        """

        new_transformer = LiveDataTransformer(
            user=user,
            transformer_name=transformer_name,
            folder_uuid=folder_uuid
        )
        # saves the files
        new_transformer.create_mod_file(transformer_folder)
        new_transformer.save()

        return new_transformer

    def transform(self, raw_features_json):
        """
            Method to transform raw features into final features with
            correct types and order.
        """
        transformer = self.load_module()
        result = eval('{}.transform(raw_features_json)'.format(transformer))
        return result

    def load_module(self):  # load_only=False):
        mod_path = '.'.join(['_dynamic_modules',
                             self.user.username,
                             self.folder_uuid,
                             self.transformer_name,
                             self.get_module_name()])
        from importlib import import_module

        loaded_mod = import_module(mod_path)
        setattr(sys.modules[__name__], self.MODULE_NAME, loaded_mod)
        return self.MODULE_NAME

    def get_user_dir(self):
        user_dir_path = os.path.join(
            self.DYNAMIC_MOD_PATH, str(
                self.user.username))
        return user_dir_path

    def get_uuid_dir(self):
        path = self.get_user_dir()
        transformer_code_dir = os.path.join(path, self.folder_uuid)
        return transformer_code_dir

    def get_module_dir(self):
        path = self.get_uuid_dir()
        module_dir = os.path.join(path, self.transformer_name)
        return module_dir

    def get_module_name(self):
        #         result = self.transformer_name + '_' + str(self.transformer_name)
        result = 'transformer'

        return result

    def get_module_filepath(self):
        result = os.path.join(self.get_module_dir(), self.get_module_name())
        return result

    def _add_init_file(self, path):
        init_file_path = os.path.join(path, '__init__.py')
        if not os.path.isfile(init_file_path):
            with open(init_file_path, 'w') as fp:
                fp.write('')
        return True

    def create_mod_file(self, transformers_path):
        """
            Attributes:
            -----------
            Creates the dir to store the loaded code from the model.
            Creates a unique uuid for each transformer
        """
        if not os.path.isdir(self.DYNAMIC_MOD_PATH):
            os.mkdir(self.DYNAMIC_MOD_PATH)
        uuid_dir = self.folder_uuid
        # path for transformer is /user_name/folder_uuid/transformer_name

        # making folder for user
        user_dir = self.get_user_dir()
        if not os.path.isdir(user_dir):
            os.mkdir(user_dir)

        # init file for user dir
        self._add_init_file(user_dir)

        # making folder_uuid folder
        uuid_dir = self.get_uuid_dir()
        if not os.path.isdir(uuid_dir):
            # init file for the transformer module
            os.mkdir(uuid_dir)

        # init file for folder uuid
        self._add_init_file(uuid_dir)

        # making transformer folder
        transformer_code_dir = self.get_module_dir()

        # saving files in final folder
        import shutil
        if isinstance(transformers_path, list):
            # list of transformers
            for transformer_path in transformers_path:
                shutil.copytree(transformer_path, transformer_code_dir)
        else:
            # directory
            shutil.copytree(transformers_path, transformer_code_dir)
        # init file for folder uuid
        self._add_init_file(transformer_code_dir)

        return transformers_path


class LiveDeploymentModel(models.Model):
    """
        Model to store a trained ML model and store it in the db.
        Attributes:
        -----------
        ml_model_name: Name given to the model to deploy.

        trained_model_file: Content(Bytes) of the trained model file obtained from calling the

        save_model() method of the estimator.

        encoder(optional): Encoder used to encode the features for the model.

        isotonic_model(optioanl): Model to normalize the output probablities based on a threshold

        probability_threshold_pass: Whether to pass based on whether the prediction is less than
        threshold or more than the threshold.

        columns:
        --------
        A binary format list of ordered columns for the model. Its a PickledObjectField so contains
        native python list.

        columns_type_mapping:
        --------------------
        A binary format dict of each column(key) and its type(value). Its a PickledObjectField so
        contains native python dict with key as column and and value as column dtype
    """
    EXPLAINABILITY_CHOICES = [
        ('shap', 'shap'),
        ('surrogate_model', 'surrogate_model')
    ]
    user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    ml_model_name = models.CharField(max_length=255)  # any meaningful name
    trained_model_file = models.BinaryField()  # bytes estimator from save_model()
    isotonic_model = models.BinaryField(null=True)
    oot_preds = PickledObjectField(null=True)
    monsoon_score_min = models.IntegerField(null=True)
    monsoon_score_max = models.IntegerField(null=True)

#     probability_threshold = models.DecimalField(
#         max_digits=12, decimal_places=10)

    raw_feature_transformer = models.ForeignKey(
        LiveDataTransformer, on_delete=models.CASCADE)
    explainability_option = models.CharField(max_length=55,
                                             choices=EXPLAINABILITY_CHOICES,
                                             null=True)
    columns = PickledObjectField()  # to subset for passing to predict_proba (List)
    columns_type_mapping = PickledObjectField()  # dtype of final training data
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_default_model(self):
        lcache = caches['localmem']
        # adding for caching
        model_cache_key = self.user.username + '_' + self.ml_model_name
        model = lcache.get(model_cache_key)  # get model from cache

        if model is None:
            model = pickle.loads(self.trained_model_file)
            lcache.set(model_cache_key, model, None)  # save in the cache
            # in above line, None is the timeout parameter. It means cache
            # forever

#         result = Estimator.load_model_bytes(self.trained_model_file)
        return model

    def get_isotonic_model(self):
        if self.isotonic_model:
            isotonic_model = NominalCalibration.load_model_bytes(
                self.isotonic_model)
            return isotonic_model
        else:
            return None

    def get_prediction(self, raw_features_df, transformed_df=None, **kwargs):
        """
            transformed_df: if its passed then no tranformer is called
        """
        #trained_model = self.get_default_model()

        transformer = self.raw_feature_transformer
        if transformed_df is None:
            final_features_df = transformer.transform(raw_features_df)
            if isinstance(final_features_df, dict):
                # errors returned by transformer
                return None, None, None,final_features_df,\
                    None, None, None, None


        else:
            final_features_df = transformed_df

        final_features = final_features_df
#         for col, dtype in self.columns_type_mapping.items():
#             if pd.notnull(final_features[col]).any():
#                 final_features[col] = final_features[col].astype(dtype)

        # generating predictions
        #prediction = trained_model.predict_proba(final_features.values)
#         if self.isotonic_model:
#             isotonic_model = self.get_isotonic_model()
#             prediction = isotonic_model.transform(prediction)
        # getting percentile
#         monsoon_score = None
#         perc = None
#         if self.oot_preds:
#             perc = self.calc_percentile(prediction.tolist()[0])
#             if self.monsoon_score_max:
#                 monsoon_score = self.get_monsoon_score(
#                     perc, self.monsoon_score_min, self.monsoon_score_max)

#         top_5 = None
#         bottom_5 = None
#         if self.explainability_option:
#             if self.explainability_option == 'shap':
#                 lcache = caches['localmem']
#                 shap_cache_key = self.user.username + '_' + self.ml_model_name + '_shap'
#                 explainers_list = lcache.get(
#                     shap_cache_key)  # get model from cache

#                 if explainers_list is None:
#                     explainers_list = shap_values.get_explainers_list(
#                         trained_model)
#                     lcache.set(
#                         shap_cache_key,
#                         explainers_list,
#                         None)  # save in the cache
#                     # in above line, None is the timeout parameter. It means
#                     # cache forever

#         #         result = Estimator.load_model_bytes(self.trained_model_file)
#                 top_5, bottom_5 = shap_values.get_top_fives_col_names(
#                     explainers_list, final_features)

#                 top_5 = top_5[['column', 'values']].to_dict(orient='records')
#                 bottom_5 = bottom_5[['column', 'values']
#                                     ].to_dict(orient='records')

#             elif self.explainability_option == 'surrogate_model':
#                 # needed for surrogate_model code, will be removed later
#                 transformed_df.loc[:, 'LAN'] = ''
#                 # creating base urls where files will be stored
#                 static_dir = static('').replace('/', '')
#                 base_url = urllib.parse.urljoin(
#                     kwargs['request_absolute_uri'].split('/api')[0], static_dir + '/')
#                 base_url = urllib.parse.urljoin(base_url, 'mle_plots/')
#                 base_url = urllib.parse.urljoin(
#                     base_url, self.ml_model_name + '/')
#                 # making path to folder that contains csvs with training data
#                 trained_data = pd.read_pickle(
#                     os.path.join(
#                         BASE_DIR,
#                         'resources',
#                         'mle_resources',
#                         self.ml_model_name,
#                         'train_df_with_flag.pkl'))
#                 # getting important reasons
#                 top_5_list, bottom_5_list = generating_most_imp_features.mle_api(
#                     transformed_df, trained_data,
#                     os.path.join(
#                         BASE_DIR,
#                         'resources',
#                         'mle_resources',
#                         self.ml_model_name),
#                     #                     base_url_for_plots=request.build_absolute_uri().split('/api')[0]+'/'+'static/mle_plots/',
#                     base_url_for_plots=base_url,
#                     save_path_for_plots=os.path.join(
#                         BASE_DIR, static_dir, 'mle_plots', self.ml_model_name),
#                     tree_folder_path=os.path.join(
#                         BASE_DIR, 'resources', 'mle_resources', self.ml_model_name)
#                 )
#                 top_5 = filter_surrogate_model_list(top_5_list)
#                 bottom_5 = filter_surrogate_model_list(bottom_5_list)

        return None, None, transformed_df, final_features, \
                None, None, None, None

    def calc_percentile(self, input_prediction):
        perc = 100 - \
            float(round(percentileofscore(self.oot_preds, input_prediction), 2))
        return round(perc, 2)

    def get_encoders(self):
        return Encoder.objects.filter(user=self.user, associated_model=self)

    def get_monsoon_score(self, perc, min_score, max_score):
        return ((perc / 100) * (max_score - min_score)) + min_score


def get_etc_ntc_classification(analysis_sample):
    acc_cols = [
        'num_accounts',
        'num_accounts_VASTUHFC',
        'num_accounts_mean_co_app',
        'num_accounts_VASTUHFC_mean_co_app']
    analysis_sample.loc[:,
                        acc_cols] = analysis_sample.loc[:,
                                                        acc_cols].fillna(0)

    if (
        (
            analysis_sample['num_accounts'] == 0).any() & (
            analysis_sample['num_accounts_VASTUHFC'] == 0).any() & (
                analysis_sample['num_accounts_mean_co_app'] == 0).any() & (
                    analysis_sample['num_accounts_VASTUHFC_mean_co_app'] == 0).any()):
        return 'ntc_ntc'

    elif ((analysis_sample['num_accounts'] == 0).any() & (analysis_sample['num_accounts_VASTUHFC'] == 0).any()) & \
        ((analysis_sample['num_accounts_mean_co_app'] != 0).any() |
            (analysis_sample['num_accounts_VASTUHFC_mean_co_app'] != 0).any()):
        return 'ntc_etc'

    elif ((analysis_sample['num_accounts'] != 0).any() | (analysis_sample['num_accounts_VASTUHFC'] != 0)).any() & \
        ((analysis_sample['num_accounts_mean_co_app'] == 0).any() &
         (analysis_sample['num_accounts_VASTUHFC_mean_co_app'] == 0).any()):
        return 'etc_ntc'

    else:
        return 'etc_etc'


def filter_surrogate_model_list(original_list):
    list_to_return = []
    for original_dict in original_list:
        dict_to_return = {}
        for k, v in original_dict.items():
            if k in ['Feature Name', 'Feature value for Sample', 'Link']:
                dict_to_return[k] = v
        list_to_return.append(dict_to_return)
    return list_to_return


class LiveDeploymentPipelineGroup(models.Model):
    #user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    pipeline_group_name = models.CharField(max_length=255)
    # user tranformer for the entire pipeline so transformer will be loaded
    # only for once
    common_transformer = models.BooleanField()
    json_type_dict = PickledObjectField(null=True)  # type dict

    def run_pipeline(self, raw_features_json, **kwargs):
        # this is called in views
        pipeline_first = LiveDeploymentModelPipeline.objects.get(
             pipeline_group=self, is_first=True)

        prediction, next_in_pipeline, transformed_df, subsetted_df, perc, monsoon_score, top_5, bottom_5 = \
            pipeline_first.live_model.get_prediction(raw_features_json, None,
                                                     request_absolute_uri=kwargs['request_absolute_uri'])
#         threshold = Threshold.objects.get(
#             user=user, ml_model=pipeline_first.live_model, is_first=True)
        next_in_pipeline = None

        while isinstance(next_in_pipeline, LiveDeploymentModelPipeline):
            if not self.common_transformer:
                # else get_prediction will not run the transformer
                transformed_df = None
            prediction, transformed_df, subsetted_df, perc, monsoon_score, top_5, bottom_5 = \
                next_in_pipeline.live_model.get_prediction(raw_features_json, transformed_df,
                                                           request_absolute_uri=kwargs['request_absolute_uri'])
            threshold = Threshold.objects.get(
                user=user, ml_model=next_in_pipeline.live_model, is_first=True)
            next_in_pipeline = threshold.get_next(prediction)

        return next_in_pipeline, prediction, subsetted_df, transformed_df, perc, monsoon_score, top_5, bottom_5


class LiveDeploymentModelPipeline(models.Model):
    """
        Attributes:
        -----------
        json_type_dict: a dict with the same structure as the json requested from client
        for the API where keys are the same however values are replaced by its python native class
        such as dict, int etc except list each list has to further desrcibed.
        However if there are list of items in any key we only need the item once.
    """
    #user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    live_model = models.ForeignKey(
        LiveDeploymentModel,
        on_delete=models.CASCADE)
    pipeline_group = models.ForeignKey(
        LiveDeploymentPipelineGroup,
        on_delete=models.CASCADE)
    is_first = models.BooleanField()  # first model in the pipeline
    # tells you if the model points to another model or returns a boolean value
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Threshold(models.Model):
    MODEL_RETURN_CHOICES = [
        ('model', 'Model'),
        ('true', True),
        ('false', False),
        ('threshold', 'threshold'),
        ('string', 'string'),
        ('break', None)
    ]

    #user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    threshold_name = models.CharField(max_length=255)
    ml_model = models.ForeignKey(LiveDeploymentModel, on_delete=models.CASCADE)
    probability_threshold = models.DecimalField(
        max_digits=12, decimal_places=10)

    # tells you if the model points to another model or returns a boolean value
    action_if_greater_prob = models.CharField(
        max_length=255, choices=MODEL_RETURN_CHOICES)
    action_if_lower_prob = models.CharField(
        max_length=255, choices=MODEL_RETURN_CHOICES)

    object_reference_if_greater = models.CharField(max_length=255)
    object_reference_if_lower = models.CharField(max_length=255)

    is_first = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_return_choice(self, return_input):
        return next(
            filter(
                lambda x: x[0] == return_input,
                self.MODEL_RETURN_CHOICES))[1]

    def get_next_object_to_run(self, next_action, next_object_reference):
        if next_action == 'Model':
            ml_model = LiveDeploymentModel.objects.get(
                ml_model_name=next_object_reference,
                user=self.user
            )
            return LiveDeploymentModelPipeline.objects.get(
                live_model=ml_model, user=self.user)
        elif next_action == 'threshold':
            return Threshold.objects.get(
                threshold_name=next_object_reference,
                user=self.user, ml_model=self.ml_model
            )
        elif next_action == 'string':
            return next_object_reference
        return next_action

    def get_next(self, prob):
        if prob is None:
            # if there is an error
            return None
        # prob is an array
        if prob[0] > self.probability_threshold:
            next_action = self.get_return_choice(self.action_if_greater_prob)
            next_action_object = self.get_next_object_to_run(
                next_action, self.object_reference_if_greater
            )
        else:
            next_action = self.get_return_choice(self.action_if_lower_prob)
            next_action_object = self.get_next_object_to_run(
                next_action, self.object_reference_if_lower
            )
        while isinstance(next_action_object, Threshold):
            # if another threshold is returned, then keep iterating
            # till another object is returned
            next_action_object = next_action_object.get_next(prob)
        return next_action_object


class Encoder(models.Model):
    user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    encoder_name = models.CharField(max_length=255)
    encoder_dict = PickledObjectField()  # bytes estimator from save_model()
    associated_model = models.ForeignKey(
        LiveDeploymentModel, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    encoder_class_name = models.CharField(max_length=255)

    def load_encoder_from_dict(self, encoding_dict):
        # making an instance of the appropriate encoder
        encoder_class = eval('encoding.' + self.encoder_class_name + '()')
        encoder_class.encoding_dict = encoding_dict
        return encoder_class

    def encode_features(self, final_features_df):
        # moved here from pipeline
        encoder = self.load_encoder_from_dict(self.encoder_dict)

        encoder.categorical_columns = list(encoder.encoding_dict.keys())
        encoded_df = encoder.transform(final_features_df)

        return encoded_df


class LiveShapImage(models.Model):
    """
        Stores Charts and Diagrams for each feature stored in featureInsight
        Each entry in this model/table stores 1 chart.
        The url can be accesed using stored_image.storage_url property

        Example:
        -------

        image = FeatureShapImage.objects.get(pk=X)
        image.stored_image.storage_url field can be used to access the image.


    """
    user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    top_5_features = PickledObjectField()
    top_5_features_name = PickledObjectField()
    bottom_5_features = PickledObjectField()
    bottom_5_features_name = PickledObjectField()


class LiveScore(models.Model):
    user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    request_id = models.CharField(max_length=255)
    request_data = PickledObjectField()
    prediction = models.DecimalField(max_digits=15, decimal_places=8)
    percentile = models.DecimalField(
        max_digits=15, decimal_places=8, null=True)
    monsoon_score = models.DecimalField(
        max_digits=15, decimal_places=8, null=True)
    approved_flag = models.CharField(max_length=255, null=True)
    shap = models.ForeignKey(
        LiveShapImage,
        on_delete=models.CASCADE,
        null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    pipeline = models.ForeignKey(
        LiveDeploymentPipelineGroup,
        on_delete=models.CASCADE)
    # used to show data in the UI table
    table_df = PickledObjectField(null=True)
