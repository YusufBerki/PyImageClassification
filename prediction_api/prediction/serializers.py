from rest_framework import serializers
from prediction.models import Prediction


class PredictionSerializer(serializers.ModelSerializer):
    class Meta():
        model = Prediction
        fields = ('file',)
