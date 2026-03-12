import cv2
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
    model = keras.models.load_model("face_multi_task_2.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    model = keras.models.load_model("face_multi_task_2.keras")

# Mapping for the Race categories (UTKFace standard)
RACE_LABELS = ["White", "Black", "Asian", "Indian", "Others"]
GENDER_LABELS = ["Male", "Female"]

def apply_clahe(image_np):
    """
    Optimizes image contrast and lighting using CLAHE.
    Input: Image as a Numpy array (RGB).
    Output: Processed Image (RGB).
    """
    # 1. Convert RGB to LAB color space
    # L = Lightness, A & B = Color components
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 2. Create the CLAHE object
    # clipLimit: higher numbers = more contrast (3.0 is a good sweet spot)
    # tileGridSize: the size of the local neighborhoods for equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # 3. Apply CLAHE only to the L-channel
    l_enhanced = clahe.apply(l)

    # 4. Merge the enhanced L-channel back with original A and B channels
    enhanced_lab = cv2.merge((l_enhanced, a, b))

    # 5. Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced_rgb


def preprocess_image(image_bytes):
    # Match the 224x224 size from your training
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))

    # Apply CLAHE to enhance contrast before inference
    img_array = np.array(img, dtype="uint8")
    img_array = apply_clahe(img_array)

    # Convert to float32 and add batch dimension
    img_array = img_array.astype("float32")
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
    age_center = round(pred_age)
    age_range = f"{max(0, age_center - 2)}-{age_center + 2}"

    # Gender: Sigmoid output (0.5 threshold)
    gender_prob = float(predictions[1][0][0])
    pred_gender = GENDER_LABELS[0] if gender_prob < 0.5 else GENDER_LABELS[1]
    gender_score = round(gender_prob if gender_prob > 0.5 else 1 - gender_prob, 2)

    # Race: Softmax output (pick highest probability)
    race_idx = np.argmax(predictions[2][0])
    pred_race = RACE_LABELS[race_idx]
    race_score = float(np.max(predictions[2][0]))

    warnings = []
    if gender_score < 0.5:
        warnings.append("Low confidence in gender prediction")
    if race_score < 0.5:
        warnings.append("Low confidence in race prediction")

    return {
        "age": age_range,
        "gender": pred_gender,
        "race": pred_race,
        "confidence": {
            "gender_score": gender_score,
            "race_score": round(race_score, 2),
        },
        **({"warnings": warnings} if warnings else {}),
    }

