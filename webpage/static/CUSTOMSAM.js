document.addEventListener("DOMContentLoaded", function() {
    var form = document.getElementById('form');
    if (!form) {
        console.error("Form element with ID 'form' not found.");
        return;
    }

    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent form submission

        // Collect form data
        var formData = {
            glucose_level: document.getElementById('Glucose').value,
            blood_pressure: document.getElementById('BloodPressure').value,
            skin_thickness: document.getElementById('SkinThickness').value,
            insulin: document.getElementById('Insulin').value,
            bmi: document.getElementById('BMI').value,
            age: document.getElementById('Age').value,
            diabetes_pedigree: document.getElementById('DiabetesPedigreeFunction').value,
            pregnancy: document.getElementById('Pregnancies').value
        };

        // Log the data being sent to the backend
        console.log("Data being sent to backend:", formData);

        fetch('/predict', { // Send POST request to "/predict" endpoint
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        })        
        
        .then(response => response.json())
        .then(data => {
            // Handle response from backend
            console.log("Data received from backend:", data);
            // Interpret prediction and display result in popup
            var predictionText = data.prediction === 1 ? "Diabetic" : "Not Diabetic";
            console.log("Popup content:", predictionText);
            document.getElementById('popupContent').textContent = ":" + predictionText;
            document.getElementById('popup').classList.add('active');
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    // Add event listener to the PREDICT button if it exists
    var predictButton = document.querySelector('.u-btn-2');
    if (predictButton) {
        predictButton.addEventListener('click', function(event) {
            event.preventDefault();
            form.dispatchEvent(new Event('submit'));
        });
    } else {
        console.error("Button with class 'u-btn-2' not found.");
    }
});

function closePopup() {
    document.getElementById('popup').classList.remove('active');
}
