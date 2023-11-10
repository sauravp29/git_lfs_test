import os

from django.test import TestCase
from .models import LiveDataTransformer, LiveDeploymentModel, LiveDeploymentPipelineGroup, \
    LiveDeploymentModelPipeline
from user_app.models import ApiUser

import pandas as pd
from client_api.settings import BASE_DIR
from monitoring_validation.validation import NUMERICAL_TYPES, CATEGORICAL_TYPES
from ml_implementation.ml_modules.ml_auto import encoding
from ml_implementation.ml_modules.ml_auto.custom_estimator import Estimator
from ml_implementation.ml_modules.ml_auto.utils import get_columns_type_dict_from_dataframe
from client_api.settings import BASE_DIR
import xgboost as xgb

TARGET = 'target'
TEST_FILE = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    'tests_data',
    'test_training.csv')
ENCODING_FILE = os.path.join(
    BASE_DIR,
    'live_scorecards',
    'tests_data',
    'test_encoder.enc')
PICKLED_MODEL = os.path.join(
    BASE_DIR,
    'live_scorecards',
    'tests_data',
    'test_trained_model.model')
TRANSFORMER_FILE = os.path.join(
    BASE_DIR,
    'live_scorecards',
    'tests_data',
    'test_transformer.py')


def _create_test_data():
    filepath = os.path.join(
        BASE_DIR,
        'live_scorecards',
        'tests_data',
        'test_training.csv')
    df = pd.read_csv(filepath)
    df = df.drop('index', axis=1)

    df_validate = df.iloc[:1]
    df[TARGET].replace('<=50K', 0, inplace=True)
    df[TARGET].replace('>50K', 1, inplace=True)

    Y = df[TARGET].values

    cat_columns = [i for i in df.columns if df[i].dtype in CATEGORICAL_TYPES]
    num_columns = [i for i in df.columns if df[i].dtype in NUMERICAL_TYPES]

    cols = cat_columns + num_columns
    if len(cols) != len(df.columns):
        raise ValueError('Invalid types in data')

    col_types = get_columns_type_dict_from_dataframe(df)

    X = df.loc[:, cols]
    le = encoding.FreqeuncyEncoding(
        categorical_columns=cat_columns, return_df=True)
    X = le.fit_transform(X, Y)

    le.save_encoding(ENCODING_FILE)

    for col in cat_columns:
        _mode = X[col].mode()
        X[col].fillna(_mode, inplace=True)

    for col in num_columns:
        _median = X[col].median()
        X[col].fillna(X[col].median(), inplace=True)

    X = X.values
    es = Estimator(xgb.XGBClassifier(), n_splits=6)
    es.fit_transform(X, Y)
    es.save_model(PICKLED_MODEL)

    return cols, col_types, df_validate


class TestTransformer(TestCase):

    def setUp(self):
        self.user = ApiUser(username='testuser', password='test_password')
        self.user.save()

    def test_transformer_run(self):
        cols, map_dict, df_validate = _create_test_data()
        with open(TRANSFORMER_FILE, 'rb') as fp:
            binary_py = fp.read()

        self.transformer = LiveDataTransformer.create(
            binary_py, 'test_transformer', self.user)
        self.transformer = LiveDataTransformer.objects.get(
            pk=self.transformer.pk)

        loaded_mod = self.transformer.load_module(False)

        self.assertTrue(getattr(loaded_mod, 'transform'))

        df = self.transformer.transform(df_validate)

    def tearDown(self):
        self.user.delete()
        self.transformer.delete()


def _create_live_model(user):
    cols, map_dict, df_validate = _create_test_data()
    with open(TRANSFORMER_FILE, 'rb') as fp:
        binary_py = fp.read()

    transformer = LiveDataTransformer.create(
        binary_py, 'test_transformer', user)

    with open(ENCODING_FILE, 'rb') as fp, open(PICKLED_MODEL, 'rb') as fp1:
        saved_model_bytes = fp1.read()
        saved_encoder_bytes = fp.read()

    live_model = LiveDeploymentModel(
        user=user,
        ml_model_name='test_model_1',
        trained_model_file=saved_model_bytes,
        encoder_file=saved_encoder_bytes,
        probability_threshold=0.78,
        columns=cols,
        columns_type_mapping=map_dict,
        probability_threshold_pass=LiveDeploymentModel.THRESHOLD_PASS_CHOICE[0][0],
        raw_feature_transformer=transformer)

    live_model.save()
    return cols, map_dict, df_validate, live_model, transformer


class TestLiveModel(TestCase):

    def setUp(self):
        self.user = ApiUser(username='testuser', password='test_password')
        self.user.save()

    def test_live_model(self):
        cols, _, df_validate, live_model, _ = _create_live_model(self.user)
        self.live_model = LiveDeploymentModel.objects.get(pk=live_model.pk)
        pred = live_model.get_prediction(df_validate)

    def tearDown(self):
        self.live_model.delete()


class TestLiveModelPipeline(TestCase):

    def setUp(self):
        self.user = ApiUser(username='testuser', password='test_password')
        self.user.save()

    def tearDown(self):
        self.user.delete()

    def test_pipeline(self):
        pipeline_group = LiveDeploymentPipelineGroup(
            user=self.user,
            pipeline_name='test_pipeline'
        )

        pipeline_group.save()

        _, _, df_validate, live_model, _ = _create_live_model(self.user)
        pipeline_step_1 = LiveDeploymentModelPipeline(
            user=self.user,
            live_model=live_model,
            pipeline_group=pipeline_group,
            is_first=True,
        )

        pipeline_step_1.save()

        pipeline_step_2 = LiveDeploymentModelPipeline(
            user=self.user,
            live_model=live_model,
            pipeline_group=pipeline_group,
            is_first=False,
        )

        pipeline_step_2.save()

        pipeline_step_1.pass_to_next_on_reject = pipeline_step_2
        pipeline_step_1.save()

        result = pipeline_group.run_pipeline(self.user, df_validate)
        explainers_list = map(shap.TreeExplainer, model.fitted_models)
