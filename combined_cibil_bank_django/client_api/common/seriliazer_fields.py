from rest_framework import serializers
import base64
import pickle


class BinaryB64Field(serializers.Field):
    """
        Seriliazes Binary File Field to b64 encoding
    """

    def to_representation(self, value):
        encoded_data = base64.b64encode(value)  # value is bytes
        return encoded_data

    def to_internal_value(self, data):
        decoded_data = base64.b64decode(data)
        return decoded_data


class PickledField(serializers.Field):
    """
        Converts b64 file uploaded to serializer to python objects
    """

    def to_representation(self, value):
        encoded_data = ''
        return encoded_data  # it is write only

    def to_internal_value(self, data):
        decoded_data = base64.b64decode(data)
        py_obj = pickle.loads(decoded_data)
        return py_obj
