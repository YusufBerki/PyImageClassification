from django.urls import path
from prediction.api.views import PredictionAPIView

urlpatterns = [

    path('', PredictionAPIView.as_view()),

]
