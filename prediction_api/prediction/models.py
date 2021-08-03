from django.db import models


class Prediction(models.Model):
    file = models.FileField()
    prediction_results = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
