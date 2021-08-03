import json

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from prediction.src.predict import predict_image
from prediction.serializers import PredictionSerializer


class PredictionAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = PredictionSerializer

    def post(self, request, *args, **kwargs):
        serializer = PredictionSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        instance = serializer.save()

        prediction_results = predict_image(instance.file.path)

        instance.prediction_results = json.dumps(prediction_results)
        instance.save()

        response = {
            "file": instance.file.url,
            "results": prediction_results,
        }
        return Response(response, status=status.HTTP_201_CREATED)
