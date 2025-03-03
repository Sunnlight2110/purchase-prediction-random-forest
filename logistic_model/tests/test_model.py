import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ..model_training import train_model

class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create sample test data
        cls.sample_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'EstimatedSalary': [50000, 60000, 70000, 80000, 90000],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Purchased': [0, 1, 1, 0, 1]
        })

    def test_model_creation(self):
        model = train_model(self.sample_data, 'Purchased')
        self.assertIsInstance(model, RandomForestClassifier)

    def test_data_preprocessing(self):
        # Test if model handles categorical data
        model = train_model(self.sample_data, 'Purchased')
        test_input = pd.DataFrame({
            'Age': [32],
            'EstimatedSalary': [65000],
            'Gender': ['Male']
        })
        test_input = pd.get_dummies(test_input)
        prediction = model.predict(test_input)
        self.assertIn(prediction[0], [0, 1])

    def test_invalid_target(self):
        with self.assertRaises(KeyError):
            train_model(self.sample_data, 'NonexistentColumn')

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            train_model(empty_df, 'Purchased')

    def test_missing_values(self):
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[0, 'Age'] = np.nan
        with self.assertRaises(ValueError):
            train_model(data_with_nan, 'Purchased')

    def test_model_performance(self):
        model = train_model(self.sample_data, 'Purchased')
        X = self.sample_data.drop('Purchased', axis=1)
        X = pd.get_dummies(X)
        y = self.sample_data['Purchased']
        score = model.score(X, y)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()
