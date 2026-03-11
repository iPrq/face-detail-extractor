import keras
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load the model you saved at the end of training
# Note: Ensure keras_hub is imported if your model uses its backbone
try:
    import keras_hub
    model = keras.models.load_model("face_multi_task.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    model = keras.models.load_model("face_multi_task.keras")

# Mapping for the Race categories (UTKFace standard)
RACE_LABELS = ["White", "Black", "Asian", "Indian", "Others"]
GENDER_LABELS = ["Male", "Female"]

def preprocess_image(image_bytes):
    # Match the 224x224 size from your training
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # Convert to array and add batch dimension
    img_array = np.array(img, dtype="float32")
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Note: If you have the Rescaling(1./255) layer INSIDE your model, 
    # you don't need to divide by 255 here.
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read and preprocess
    contents = await file.read()
    img_tensor = preprocess_image(contents)

    # 2. Run Inference
    # Model returns a list: [age_output, gender_output, race_output]
    predictions = model.predict(img_tensor)
    
    # 3. Process outputs
    # Age: Multiply by 100 since we divided by 100 in DataLoader
    pred_age = float(predictions[0][0][0] * 100)
    
    # Gender: Sigmoid output (0.5 threshold)
    gender_prob = float(predictions[1][0][0])
    pred_gender = GENDER_LABELS[0] if gender_prob < 0.5 else GENDER_LABELS[1]
    
    # Race: Softmax output (pick highest probability)
    race_idx = np.argmax(predictions[2][0])
    pred_race = RACE_LABELS[race_idx]

    return {
        "age": round(pred_age, 1),
        "gender": pred_gender,
        "race": pred_race,
        "confidence": {
            "gender_score": round(gender_prob if gender_prob > 0.5 else 1 - gender_prob, 2),
            "race_score": float(np.max(predictions[2][0]))
        }
    }

