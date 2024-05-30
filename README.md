# Diabetes Detection using Support Vector Machine (SVM)

![Diabetes Detection](webpage/images/Screenshot%20(297).png)

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
![File Structure](webpage/images/Screenshot(295).png)

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

##Sample data:
![Directory Structure](webpage/images/Screenshot%20(296).png)



## Results

The model achieved an accuracy of 78.66% on the training data and 77.27% on the test data, indicating a reliable performance for diabetes prediction.


## Acknowledgements

- The dataset is sourced from the Pima Indians Diabetes Database.
- Special thanks to the developers of scikit-learn and FastAPI for their excellent libraries.
