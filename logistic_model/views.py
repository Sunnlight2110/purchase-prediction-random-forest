from django.shortcuts import render
from .model_training import ModelTrainer
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from django.conf import settings

class PurchasePredictionAPI(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_trainer = ModelTrainer()
        self.model_path = os.path.join(settings.BASE_DIR, 'saved_models')
    
    def post(self, request):
        """Handle different model operations based on action parameter"""
        action = request.data.get('action')
        
        if action == 'train':
            return self.train_purchase_predictor(request.data)
        elif action == 'optimize':
            return self.optimize_purchase_predictor(request.data)
        elif action == 'predict':
            return self.predict_purchase(request.data)
        else:
            return Response(
                {'error': 'Invalid action. Use train, optimize, or predict'},
                status=status.HTTP_400_BAD_REQUEST
            )

    def _ensure_model_ready(self):
        """Helper method to ensure model is ready for use"""
        try:
            if self.model_trainer.model is None:
                self.model_trainer.load_model(self.model_path)
            return True
        except FileNotFoundError:
            return False

    def get(self, request):
        """Get model evaluation metrics and current parameters"""
        if not self._ensure_model_ready():
            return Response(
                {'error': 'Model not found. Please train the model first using POST /model_operations/ with action="train"',
                 'status': 'untrained'},
                status=status.HTTP_404_NOT_FOUND
            )
            
        try:
            metrics = self.model_trainer.evaluate()
            
            # Handle both cases: loaded model and trained model with test data
            if metrics['model_status'] == 'loaded':
                response_data = {
                    'model_status': 'loaded',
                    'model_parameters': metrics['parameters'],
                    'feature_names': metrics['feature_names']
                }
            else:
                response_data = {
                    'model_status': 'trained',
                    'model_parameters': {
                        'n_estimators': self.model_trainer.n_estimators,
                        'max_depth': self.model_trainer.max_depth,
                        'random_state': self.model_trainer.random_state
                    },
                    'evaluation_metrics': {
                        'accuracy': float(metrics['accuracy']),
                        'cross_val_score': float(metrics['cross_val_score']),
                        'roc_auc_score': float(metrics['roc_auc_score']),
                        'classification_report': metrics['classification_report'],
                        'feature_importance': metrics['feature_importance'].to_dict('records')
                    }
                }
            
            return Response(response_data)
            
        except FileNotFoundError:
            return Response(
                {'error': 'Model not found. Please train the model first.'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            print('Error:', str(e))
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    def train_purchase_predictor(self, data):
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

    def optimize_purchase_predictor(self, data):
        try:
            if not self._ensure_model_ready():
                return Response(
                    {'error': 'Model not found. Please train the model first using action="train"'},
                    status=status.HTTP_404_NOT_FOUND
                )
                
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
            metrics = self.model_trainer.evaluate()
            print(metrics['model_status'])
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    def predict_purchase(self, data):
        try:
            # Check if model is ready
            if not self._ensure_model_ready():
                return Response(
                    {'error': 'Model not found. Please train the model first using the /model_operations/ endpoint with action="train"'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get features from request
            features = data.get('features')
            if not features or not isinstance(features, dict):
                return Response(
                    {'error': 'Invalid features format. Expected dictionary of feature values'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Age validation - return 0 if age < 20
            try:
                age = float(features.get('Age', 0))
                if age < 20:
                    return Response({
                        'prediction': 0,
                        'input_received': features,
                        'note': 'Automatic rejection: Age must be 20 or above'
                    })
            except (TypeError, ValueError):
                return Response(
                    {'error': 'Invalid age value'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Convert to DataFrame and handle missing values
            input_data = pd.DataFrame([features])
            
            # Ensure all values are numeric
            for column in input_data.columns:
                input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
            
            # Check for NaN values
            if input_data.isnull().any().any():
                return Response(
                    {'error': 'Invalid numeric values in features'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Make prediction
            prediction = self.model_trainer.predict_new(input_data)
            
            return Response({
                'prediction': int(prediction[0]),
                'input_received': features
            })
            
        except Exception as e:
            return Response(
                {'error': f'Prediction error: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

