from django.shortcuts import render,redirect

# Create your views here.
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

from disease.models import ImageUpload


IMAGE_SIZE = 256
CHANNELS = 3

MODEL_PATH = "models/4/model.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Apple scab", "Black Rot", "Cedar Apple Rust", "Healthy"]
print(CHANNELS)


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

# Define a function to compute a loss-like measure (e.g., Euclidean distance)
def compute_loss(prediction_vector, reference_vector):
    return np.sqrt(np.sum(np.square(prediction_vector - reference_vector)))

class PredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def get(self,request, format = None):
        print(CHANNELS)
        return Response({"message": "Will not appear in schema!"})
    
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['file']
        image = read_file_as_image(file_obj.read())
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        index = np.argmax(predictions[0])
        
        # loss
        prediction_vector = predictions[0]
        reference_vector = np.zeros_like(prediction_vector) 
         # Compute a loss or distance
        loss = compute_loss(prediction_vector, reference_vector)
        
        predicted_class = CLASS_NAMES[index]
        confidence = np.max(predictions[0])
        
                    # Save the image and prediction to the database
        image_prediction = ImageUpload(
                image=file_obj,
                predicted_class=predicted_class,
                confidenceforpre = float(confidence),
            )
        image_prediction.save()

        return Response({
            "message": "Prediction received and processed",
            "class": predicted_class,
            "confidence": float(confidence),
            "loss": float(loss)  
        }, status=status.HTTP_200_OK)