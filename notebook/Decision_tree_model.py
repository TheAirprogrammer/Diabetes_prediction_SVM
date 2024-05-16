import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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
scaler.fit(X)
X = scaler.transform(X)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Model
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, Y_train)

# Model Evaluation
# Accuracy score on the training data
training_data_accuracy = accuracy_score(classifier.predict(X_train), Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# Accuracy score on the test data
test_data_accuracy = accuracy_score(classifier.predict(X_test), Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# Saving the trained model
filename = 'Diabetes_decision_tree_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(classifier, file)
