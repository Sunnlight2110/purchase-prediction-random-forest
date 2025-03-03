from django.urls import path
from .views import ModelView

urlpatterns = [
    path('purchase-predictor/', ModelView.as_view(), name='purchase_prediction_operations'),
]
