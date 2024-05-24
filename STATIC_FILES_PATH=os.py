import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import joblib


# Get the absolute path of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))
print("Current directory:", current_dir)

# Construct the absolute path for the templates directory
TEMPLATES_PATH = os.path.join(current_dir, 'webpage', 'templates')
print("index page path:", TEMPLATES_PATH)

STATIC_FILES_PATH = os.path.join(current_dir, 'webpage', 'static')
print("static page path:", STATIC_FILES_PATH)

MODEL_PATH=os.path.join(current_dir, 'api')
print("smodel path:", MODEL_PATH)


# Load the model
model = joblib.load(os.path.join(MODEL_PATH, "Diabetes_model.joblib"), "r")

