import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.tree import plot_tree
import numpy as np
from model_training import RandomForestPurchasePredictorModel
import pandas as pd

class ModelVisualizer:
    def __init__(self, model_trainer: RandomForestPurchasePredictorModel):
        self.trainer = model_trainer
        plt.style.use('seaborn-v0_8-dark')  # Updated style name

    def plot_roc_curve(self):
        """Plot ROC curve for model performance"""
        if self.trainer.model is None:
            raise ValueError("Model must be trained first")

        y_pred_proba = self.trainer.model.predict_proba(self.trainer.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.trainer.y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def plot_feature_importance(self):
        """Visualize feature importance"""
        if self.trainer.model is None:
            raise ValueError("Model must be trained first")

        importance_df = pd.DataFrame({
            'Feature': self.trainer.feature_names,
            'Importance': self.trainer.model.feature_importances_
        }).sort_values(by='Importance', ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance in Random Forest Model')
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        """Plot confusion matrix as heatmap"""
        if self.trainer.model is None:
            raise ValueError("Model must be trained first")

        y_pred = self.trainer.model.predict(self.trainer.X_test)
        cm = self.trainer.evaluate_model_performance()['confusion_matrix']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_learning_curves(self):
        """Plot learning curves to show model performance with varying training data size"""
        if self.trainer.model is None:
            raise ValueError("Model must be trained first")

        train_sizes, train_scores, test_scores = learning_curve(
            self.trainer.model, 
            self.trainer.X_train, 
            self.trainer.y_train,
            cv=5,
            n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def plot_single_tree(self, tree_index=0, max_depth=3):
        """Plot a single decision tree from the random forest"""
        if self.trainer.model is None:
            raise ValueError("Model must be trained first")

        plt.figure(figsize=(20,10))
        plot_tree(self.trainer.model.estimators_[tree_index],
                 feature_names=self.trainer.feature_names,
                 max_depth=max_depth,
                 filled=True,
                 rounded=True)
        plt.title(f'Decision Tree #{tree_index} from Random Forest')
        plt.show()

    def plot_all_visualizations(self):
        """Generate all available visualizations"""
        self.plot_roc_curve()
        self.plot_feature_importance()
        self.plot_confusion_matrix()
        self.plot_learning_curves()
        self.plot_single_tree()

# Example usage:
if __name__ == "__main__":
    # Load and prepare data
    data = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\Social_Network_Ads.csv')
    data.drop(['User ID'], axis=1, inplace=True)
    
    # Train model
    trainer = RandomForestPurchasePredictorModel()
    trainer.train_purchase_model(data, 'Purchased')
    
    # Create visualizations
    visualizer = ModelVisualizer(trainer)
    visualizer.plot_all_visualizations()
