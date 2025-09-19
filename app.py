from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model + labels once at startup
model = tf.keras.models.load_model("model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Food Recognition API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((224, 224))   # Teachable Machine default
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        return JSONResponse({
            "food": labels[predicted_idx],
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})
