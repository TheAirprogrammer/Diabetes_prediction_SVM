import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# Data Collection and Analysis
diabetes_dataset = pd.read_csv('data/diabetes.csv')
print(diabetes_dataset['Outcome'].value_counts())

# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)

# Training the Model with GridSearchCV for Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}

grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_

# Using the best parameters to train the model
classifier = XGBClassifier(**best_params, random_state=42)
classifier.fit(X_train, Y_train)

# Model Evaluation
# Accuracy score on the training data
training_data_accuracy = accuracy_score(classifier.predict(X_train), Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# Accuracy score on the test data
test_data_accuracy = accuracy_score(classifier.predict(X_test), Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# Saving the trained model
filename = 'Diabetes_xgboost_tuned_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(classifier, file)
