# Diabetes Detection using Support Vector Machine (SVM)

![Diabetes Detection](webpage/images/diabetes_detection_banner.png)

## Project Description

This project is a web-based application designed to detect diabetes using a Support Vector Machine (SVM) model. The application allows users to input medical data through an interactive web interface and provides a prediction on whether the person is diabetic or not.

## Features

- **FastAPI Backend:** Efficient handling of prediction requests.
- **Pre-trained SVM Model:** Ensures high accuracy in predictions.
- **User-friendly Interface:** HTML form for easy data input.
- **Real-time Predictions:** JSON responses with clear results.

## Model Performance

- **Training Accuracy:** 78.66%
- **Testing Accuracy:** 77.27%

## File Structure
DIABETES_PREDICTION_SVM
│   .env
│   main.py
│   Procfile
├───api
│   │   diabetes_api.py
│   │   Diabetes_model.joblib
│   └───__pycache__
├───data
│       diabetes.csv
├───models
│       Diabetes_decision_tree_model.pkl
│       Diabetes_model.joblib
│       Diabetes_neural_network_model.pkl
│       Diabetes_random_forest_model.pkl
│       Diabetes_xgboost_tuned_model.pkl
├───notebook
│       Blood_suagar_prediction_model.py
│       Blood_sugar_prediction_V2.py
│       Decision_tree_model.py
│       Gradient_boosting_model.py
│       neural_network_model.py
│       Project_3_Diabetes_Prediction.ipynb
│       Random_forest_model.py
├───webpage
│   ├───images
│   ├───intlTelInput
│   ├───static
│   └───templates
└───__pycache__

## Usage
1. **Fill in the medical data in the provided form.**
2. **Click on the predict button to get the prediction result.**


## API Endpoints

- **GET /**: Renders the HTML form for input.
- **POST /predict**: Accepts JSON data and returns the prediction result.

## Model Details

- **Model Type:** Support Vector Machine (SVM)
- **Libraries Used:** scikit-learn, joblib

## Dataset

The dataset used for training the model is the Pima Indians Diabetes Database, which includes the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome

Sample data:
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0

## Results

The model achieved an accuracy of 78.66% on the training data and 77.27% on the test data, indicating a reliable performance for diabetes prediction.


## Acknowledgements

- The dataset is sourced from the Pima Indians Diabetes Database.
- Special thanks to the developers of scikit-learn and FastAPI for their excellent libraries.
