import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

class RandomForestPurchasePredictorModel:
    """
    Random Forest Classifier for predicting purchase behavior based on demographic data.
    Handles training, optimization, and evaluation of purchase prediction model.
    """
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

    def _validate_input_data(self, data: pd.DataFrame, target: str):
        """Validates input data for purchase prediction model"""
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if data.isnull().values.any():
            raise ValueError("Data contains missing values")
        if target not in data.columns:
            raise KeyError(f"Target column '{target}' not found in data")

    def train_purchase_model(self, data: pd.DataFrame, target: str) -> RandomForestClassifier:
        """Trains the purchase prediction model with demographic data"""
        self._validate_input_data(data, target)
        
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

    def optimize_model_parameters(self, param_grid=None):
        """Optimizes model hyperparameters for better purchase predictions"""
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

    def evaluate_model_performance(self):
        """Evaluates purchase prediction model performance metrics"""
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

# Example usage:
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\Social_Network_Ads.csv')
    data.drop(['User ID'], axis=1, inplace=True)
    
    # Initialize and train model
    trainer = RandomForestPurchasePredictorModel()
    trainer.train_purchase_model(data, 'Purchased')
    
    # Optimize parameters
    trainer.optimize_model_parameters()
    
    # Get evaluation metrics
    metrics = trainer.evaluate_model_performance()
    print("Model Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"\n{metric_name}:")
        print(value)