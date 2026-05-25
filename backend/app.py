import os
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_image
from gradcam import generate_gradcam

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "../uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():

    return {
        "message": "Breast Cancer Detection API"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_image(file_path)

    gradcam_path = generate_gradcam(file_path)

    return {
        "prediction": result["class"],
        "confidence": result["confidence"],
        "gradcam": gradcam_path
    }