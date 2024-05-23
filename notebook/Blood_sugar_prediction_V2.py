import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Data Collection and Analysis
diabetes_dataset = pd.read_csv('data/diabetes.csv')
print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset.groupby('Outcome').mean())

# Separating the data and labels
X = diabetes_dataset.drop(columns=['Outcome'], axis=1)
Y = diabetes_dataset['Outcome']

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Hyperparameter Tuning with GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Training the Model with Best Parameters
classifier = SVC(**best_params)
classifier.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)

# Feature Selection with RFE
selector = RFE(estimator=SVC(kernel='linear'), n_features_to_select=5)
selector = selector.fit(X_train, Y_train)

selected_features = X.columns[selector.support_]
print("Selected Features:", selected_features)

# Retraining the Model with Selected Features
X_train_selected = X[selected_features]
X_test_selected = X_test[selected_features]

classifier.fit(X_train_selected, Y_train)

# Model Evaluation with Selected Features
X_train_prediction_selected = classifier.predict(X_train_selected)
training_data_accuracy_selected = accuracy_score(X_train_prediction_selected, Y_train)
print('Accuracy score of the training data with selected features:', training_data_accuracy_selected)

X_test_prediction_selected = classifier.predict(X_test_selected)
test_data_accuracy_selected = accuracy_score(X_test_prediction_selected, Y_test)
print('Accuracy score of the test data with selected features:', test_data_accuracy_selected)

# Saving the Model
filename = 'Diabetes_model.joblib'
pickle.dump(classifier, open(filename, 'wb'))
