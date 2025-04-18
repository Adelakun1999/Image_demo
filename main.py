from pydantic import BaseModel
from PIL import Image
import os 
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import io
from torchvision import transforms
import torch
import torchvision
import requests
from torch import nn
import uvicorn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification



image_class_names = ['NOT SAFE', 'SAFE', 'QUESTIONABLE']
text_class_names = ['NOT SAFE', 'SAFE']


weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT

loaded_model = torchvision.models.efficientnet_b0(weights=weight)

loaded_model.classifier = nn.Linear(in_features=1280, out_features=len(image_class_names))

loaded_model.load_state_dict(torch.load(f='nudex.pth', weights_only=True,  map_location=torch.device('cpu')))
loaded_model.eval()


checkpoint = "distilbert-base-cased"


loaded_model_text = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                                  
                                                              num_labels = 2)

loaded_model_text.load_state_dict(torch.load(f='nude.pth', map_location=torch.device('cpu'), weights_only=True))
loaded_model_text.eval()


transforms = weight.transforms()


app = FastAPI(
     title='SingleEye Image classification',
     description=(
        "This API allows users to upload an image (in ,jpg, JPEG or PNG format) "
        "for classification into one of three categories: SAFE, NOT SAFE, or QUESTIONABLE."),
         version="1.0.0")

@app.get('/', 
         tags=["General"],
         summary="API Health Check",
         description="Returns a simple message to confirm the API is running.")


def get():
    return {'Message ' : 'Single eye Image classification'}



# Pydantic model for Url input  

class ImageURL(BaseModel):
    image_url : str
    text_content : str 

    model_config = {
        "json_schema_extra" : {
            "example" : {
                "text_content" : "Write any text here" ,
                "image_url" : "https://example.com/image.jpg"
            }
        }
    }



#Endpoint for URL-based Image Prediction

def preprocessing(text): 
    checkpoint = "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_word = tokenizer(text, return_tensors='pt')
    
    return tokenized_word

@app.post("/predict-image-url",
           tags=["Image Classification/Text Classification"],
        summary="Classify an Image",
        description=(
        "Upload the image url to classify it as "
        "'SAFE', 'NOT SAFE', or 'QUESTIONABLE'. The image will be preprocessed and "
        "evaluated by the machine learning model, which outputs the predicted label."
    ),
    response_description="The predicted label for the uploaded image." )


async def predict_image_url(image_url: ImageURL):
    try:
        # Fetch the image from the provided URL
        response = requests.get(image_url.image_url)
        response.raise_for_status()  # Ensure the request was successful
        
        # Load the image into PIL
        image = Image.open(io.BytesIO(response.content))
        image = image.convert('RGB')
        
        # Preprocess the image and add a batch dimension
        input_tensor = transforms(image).unsqueeze(dim=0)

        # Run inference
        with torch.inference_mode():
            y_logit = loaded_model(input_tensor)
            probabilities = y_logit.softmax(dim=1)

        confidence , predicted_class = probabilities.max(dim=1)
        
        #for the text content 
        text = image_url.text_content
        tokenized_word = preprocessing(text)

        with torch.inference_mode():
            y_logit_text = loaded_model_text(**tokenized_word)
            y_logit_text = y_logit_text.logits
            probabilities_text = y_logit_text.softmax(dim=1)


        confidence_text , predicted_class_text = probabilities_text.max(dim=1)
        
        
        
        return {"image_url" : image_url.image_url,
                'text_content' : image_url.text_content,
                'Predicted_Class_Image' : image_class_names[predicted_class.item()],
                'predicted_class_text' : text_class_names[predicted_class_text.item()],
                'Confidence_Image' : round(confidence.item() ,3),
                'Confidence_Text' : round(confidence_text.item() ,3)}
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"},
            status_code=500
        )
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
