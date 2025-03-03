from django.shortcuts import render
from .model_training import RandomForestPurchasePredictorModel
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from django.conf import settings

class PurchasePredictionAPI(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_trainer = RandomForestPurchasePredictorModel()
        self.model_path = os.path.join(settings.BASE_DIR, 'saved_models')
    
    def post(self, request):
        """Handle different model operations based on action parameter"""
        action = request.data.get('action')
        
        if action == 'train':
            return self.train_model(request.data)
        elif action == 'optimize':
            return self.optimize_model(request.data)
        elif action == 'predict':
            return self.predict(request.data)
        else:
            return Response(
                {'error': 'Invalid action. Use train, optimize, or predict'},
                status=status.HTTP_400_BAD_REQUEST
            )

    def train_model(self, data):
        try:
            # Expecting CSV file path or data in request
            training_data = pd.read_csv(data.get('data_path'))
            target_column = data.get('target_column')
            
            # Train the model
            self.model_trainer.train(training_data, target_column)
            
            # Save the model
            save_info = self.model_trainer.save_model(self.model_path)
            
            # Get evaluation metrics
            metrics = self.model_trainer.evaluate()
            
            return Response({
                'message': 'Model trained successfully',
                'model_path': save_info,
                'metrics': metrics
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    def optimize_model(self, data):
        try:
            # Load model if not already trained
            if self.model_trainer.model is None:
                self.model_trainer.load_model(self.model_path)
            
            # Get custom parameter grid if provided
            param_grid = data.get('param_grid', None)
            
            # Optimize model
            self.model_trainer.optimize_parameters(param_grid)
            
            # Save optimized model
            save_info = self.model_trainer.save_model(self.model_path)
            
            # Get new evaluation metrics
            metrics = self.model_trainer.evaluate()
            
            return Response({
                'message': 'Model optimized successfully',
                'best_parameters': {
                    'n_estimators': self.model_trainer.n_estimators,
                    'max_depth': self.model_trainer.max_depth
                },
                'metrics': metrics
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    def predict(self, data):
        try:
            # Load model if not already loaded
            if self.model_trainer.model is None:
                self.model_trainer.load_model(self.model_path)
            
            # Convert input data to DataFrame
            input_data = pd.DataFrame([data.get('features')])
            
            # Make prediction
            prediction = self.model_trainer.predict_new(input_data)
            
            return Response({
                'prediction': int(prediction[0])
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

