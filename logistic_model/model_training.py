import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import os
import joblib


class ModelTrainer:
    def __init__(self, n_estimators=300, max_depth=5, random_state=69):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def _validate_data(self, data: pd.DataFrame, target: str):
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if data.isnull().values.any():
            raise ValueError("Data contains missing values")
        if target not in data.columns:
            raise KeyError(f"Target column '{target}' not found in data")

    def train(self, data: pd.DataFrame, target: str) -> RandomForestClassifier:
        """
        Train the Random Forest model with current parameters
        """
        self._validate_data(data, target)
        
        X = data.drop(target, axis=1)
        X = pd.get_dummies(X)
        self.feature_names = X.columns
        y = data[target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def optimize_parameters(self, param_grid=None):
        """
        Find optimal parameters using GridSearchCV
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be trained first")
            
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15]
            }
        
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state),
            param_grid=param_grid,
            cv=5
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Update model parameters
        self.n_estimators = grid_search.best_params_['n_estimators']
        self.max_depth = grid_search.best_params_['max_depth']
        
        # Create new model with optimal parameters
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        # Fit the model with existing scaled data
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        """
        Evaluate the model performance
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        y_pred = self.model.predict(self.X_test)
        
        evaluation_metrics = {
            'accuracy': metrics.accuracy_score(self.y_test, y_pred),
            'confusion_matrix': metrics.confusion_matrix(self.y_test, y_pred),
            'classification_report': metrics.classification_report(self.y_test, y_pred),
            'cross_val_score': cross_val_score(self.model, self.X_train, self.y_train, cv=5).mean(),
            'roc_auc_score': metrics.roc_auc_score(self.y_test, y_pred),
            'feature_importance': pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
        }
        
        return evaluation_metrics

    def save_model(self, directory_path):
        """
        Save the trained model and scaler to files
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        model_path = os.path.join(directory_path, 'random_forest_model.joblib')
        scaler_path = os.path.join(directory_path, 'scaler.joblib')
        feature_names_path = os.path.join(directory_path, 'feature_names.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, feature_names_path)
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'feature_names_path': feature_names_path
        }
    
    def load_model(self, directory_path):
        """
        Load the trained model and scaler from files
        """
        model_path = os.path.join(directory_path, 'random_forest_model.joblib')
        scaler_path = os.path.join(directory_path, 'scaler.joblib')
        feature_names_path = os.path.join(directory_path, 'feature_names.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_names_path]):
            raise FileNotFoundError("Model files not found in specified directory")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_names_path)
        
    def predict_new(self, data: pd.DataFrame):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model must be loaded or trained first")
            
        # Ensure data has the same features as training data
        data = pd.get_dummies(data)
        missing_cols = set(self.feature_names) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        data = data[self.feature_names]
        
        # Scale the data
        scaled_data = self.scaler.transform(data)
        return self.model.predict(scaled_data)

