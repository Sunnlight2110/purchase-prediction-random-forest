#Purchase Prediction using Random Forest
This project uses the Random Forest Classifier to predict purchase behavior based on demographic data, such as age, estimated salary, and gender. The model is trained on historical data, optimized using hyperparameter tuning, and evaluated based on key performance metrics. It aims to predict whether an individual is likely to make a purchase, providing insights that can be used for targeted marketing in e-commerce.

Features
Data Preprocessing: Cleans and prepares the data by handling missing values, scaling numerical features, and encoding categorical variables.
Model Training: Trains a Random Forest model with 300 estimators and a max depth of 5, splitting data into training and testing sets.
Model Optimization: Utilizes Grid Search to tune hyperparameters for optimal performance.
Model Evaluation: Assesses the model using accuracy, confusion matrix, ROC-AUC score, and feature importance.
Visualizations: Generates key visualizations, including ROC-AUC curves, confusion matrix, and feature importance graphs.
Installation
Clone the repository:
sh
git clone https://github.com/Sunnlight2110/purchase-prediction-random-forest.git
Navigate to the project directory:
sh
cd purchase-prediction-random-forest
Install the required dependencies:
sh
pip install -r requirements.txt
Usage
Prepare the dataset and place it in the data directory.
Run the preprocessing script:
sh
python preprocess.py
Train the model:
sh
python train.py
Evaluate the model:
sh
python evaluate.py
Generate visualizations:
sh
python visualize.py
Results
Accuracy: [Insert accuracy score]
Confusion Matrix: [Include confusion matrix image or description]
ROC-AUC Score: [Insert ROC-AUC score]
Feature Importance: [Include feature importance graph]
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License.
