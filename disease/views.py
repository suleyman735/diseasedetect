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
        predicted_class = CLASS_NAMES[index]
        confidence = np.max(predictions[0])

        return Response({
            "message": "Prediction received and processed",
            "class": predicted_class,
            "confidence": float(confidence)
        }, status=status.HTTP_200_OK)