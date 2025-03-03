from django.urls import path
from .views import PurchasePredictionAPI

urlpatterns = [
    path('purchase-predictor/', PurchasePredictionAPI.as_view(), name='purchase_prediction_operations'),
]
