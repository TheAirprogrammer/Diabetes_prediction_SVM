from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import joblib
import os

# Get the absolute path of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))
print("Current directory:", current_dir)

# Construct the absolute path for the templates directory
TEMPLATES_PATH = os.path.join(current_dir, 'webpage', 'templates')
print("index page path:", TEMPLATES_PATH)

STATIC_FILES_PATH = os.path.join(current_dir, 'webpage', 'static')
print("static page path:", STATIC_FILES_PATH)

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory=STATIC_FILES_PATH), name="static")
MODEL_PATH=os.path.join(current_dir, 'api')
print("smodel path:", MODEL_PATH)



# Load the model
model = joblib.load(os.path.join(MODEL_PATH, "Diabetes_model.joblib"), "r")


class PredictionRequest(BaseModel):
    glucose_level: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    age: float
    diabetes_pedigree: float
    pregnancy: float

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(TEMPLATES_PATH, "index.html"), "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        features = [[
            request.glucose_level,
            request.blood_pressure,
            request.skin_thickness,
            request.insulin,
            request.bmi,
            request.age,
            request.diabetes_pedigree,
            request.pregnancy
        ]]
        print("Received features:", features)
        prediction = model.predict(features)
        print("Prediction:", prediction)
        
        # Convert prediction to human-readable format
        prediction_result = "The person is Diabetic" if prediction[0] == 1 else "The person is Non-Diabetic"
        
        return {"prediction": prediction_result}
    except Exception as e:
        # Handle any exceptions gracefully
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
