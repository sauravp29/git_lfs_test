from rest_framework import serializers
from .models import LiveDataTransformer, LiveDeploymentModel, LiveDeploymentModelPipeline, LiveDeploymentPipelineGroup, LiveShapImage, LiveScore
from user_app.models import ApiUser
# ?from client_data.serializers import StoredBlobSerializer
from common.seriliazer_fields import BinaryB64Field


class LiveDataTransformerSerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(
        default=serializers.CurrentUserDefault()
    )
    binary_transformer = BinaryB64Field()

    class Meta:
        model = LiveDataTransformer
        fields = '__all__'


class LiveDataTransformerSerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(
        default=serializers.CurrentUserDefault()
    )
    trained_model_file = BinaryB64Field()
    encoder_file = BinaryB64Field()
    raw_feature_transformer = serializers.PrimaryKeyRelatedField(
        queryset=LiveDataTransformer.objects.all()
    )

    class Meta:
        model = LiveDeploymentModel
        fields = '__all__'

# class LiveShapImageSerializer(serializers.ModelSerializer):
#    user = serializers.PrimaryKeyRelatedField(read_only=True)
#    stored_image = StoredBlobSerializer()
#    class Meta:
#        model = LiveShapImage
#        fields = '__all__'


class LiveScoreSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)
    #shap = LiveShapImageSerializer()

    class Meta:
        model = LiveScore
        exclude = ['request_data']
