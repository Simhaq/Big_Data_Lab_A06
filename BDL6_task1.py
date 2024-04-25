import uvicorn
from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

def Load_Model(path: str):
    """
    Load the Keras Sequential model from the given path.
    """
    model = load_model(path)
    return model

def predict_digit(model, data_point):
    """
    Predict the digit using the loaded model.
    """
    # Normalize the data
    data_point = data_point / 255.0
    # Perform prediction
    prediction = model.predict(data_point)
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)
    return str(predicted_digit)

model_path = None

@app.post('/predict')
async def predict(upload_file: UploadFile = File(...)):
    """
    Predict the digit from the uploaded image file.
    """
    # Check if model is loaded
    if model_path is None:
        return {"error": "Model not loaded. Please load the model first."}
    
    # Read the uploaded image file
    contents = await upload_file.read()
    
    img = Image.open(io.BytesIO(contents)).convert('L')  # Load the image and convert to grayscale 
    img = img.resize((28, 28))  # Resize to 28x28
    
    # Convert image to numpy array
    img_array = np.array(img)

    # Reshape the image to match the input shape of the model
    img_array = img_array.reshape(1,784)
    
    # Call predict_digit function to get the predicted digit
    digit = predict_digit(Load_Model(model_path), img_array)

    
    return {"Digit" : digit}
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python app.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]

    uvicorn.run(app)
