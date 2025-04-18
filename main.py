from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import transforms as T
import torchvision
import torch
import requests
import io
import os
from functools import lru_cache

# Class Names
image_class_names = ['NOT SAFE', 'SAFE', 'QUESTIONABLE']
text_class_names = ['NOT SAFE', 'SAFE']
checkpoint = "distilbert-base-cased"

# FastAPI App
app = FastAPI(
    title='SingleEye Image classification',
    description=(
        "Upload an image (JPG/PNG) or URL for classification into SAFE, NOT SAFE, or QUESTIONABLE."
    ),
    version="1.0.0"
)

# Health Check
@app.get("/", tags=["General"])
def root():
    return {"message": "SingleEye Image Classification API is running."}

# ---------- MODEL LOADERS WITH LRU CACHE ----------

@lru_cache()
def get_image_model():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    model.classifier = nn.Linear(in_features=1280, out_features=len(image_class_names))
    model.load_state_dict(torch.load("nudex.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model, weights.transforms()

@lru_cache()
def get_text_model():
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.load_state_dict(torch.load("nude.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model

@lru_cache()
def get_tokenizer():
    return AutoTokenizer.from_pretrained(checkpoint)

# ---------- IMAGE UPLOAD ENDPOINT ----------

@app.post("/predict-image", tags=["Image Classification"])
async def predict_image(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            return JSONResponse(content={"error": "Only JPEG and PNG images are allowed."}, status_code=400)

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model, transform = get_image_model()
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_class = probabilities.max(dim=1)

        return {
            "Predicted Class": image_class_names[predicted_class.item()],
            "Confidence": round(confidence.item(), 3)
        }

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)

# ---------- PREDICT FROM URL + TEXT ----------

class ImageURL(BaseModel):
    image_url: str
    text_content: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "text_content": "Write any text here",
                "image_url": "https://example.com/image.jpg"
            }
        }
    }

@app.post("/predict-image-url", tags=["Image Classification"])
async def predict_image_url(payload: ImageURL):
    try:
        # Load and process image
        response = requests.get(payload.image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        model, transform = get_image_model()
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

        conf_img, pred_img = probabilities.max(dim=1)

        # Process text
        tokenizer = get_tokenizer()
        tokenized = tokenizer(payload.text_content, return_tensors="pt")

        model_text = get_text_model()
        with torch.no_grad():
            output = model_text(**tokenized)
            probabilities_text = torch.softmax(output.logits, dim=1)

        conf_txt, pred_txt = probabilities_text.max(dim=1)

        return {
            "image_url": payload.image_url,
            "text_content": payload.text_content,
            "Predicted_Class_Image": image_class_names[pred_img.item()],
            "Predicted_Class_Text": text_class_names[pred_txt.item()],
            "Confidence_Image": round(conf_img.item(), 3),
            "Confidence_Text": round(conf_txt.item(), 3)
        }

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)
