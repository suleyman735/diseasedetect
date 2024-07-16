import os
from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from keras.layers import TFSMLayer
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 256
CHANNELS = 3



app = FastAPI()
MODEL_PATH = "../models/4/model.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
# MODEL = TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
# MODEL = tf.keras.models.load_model("../models/1/")
CLASS_NAMES = ["Apple scab","Black Rot","Cedar Apple Rust", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, World"


def read_file_as_image(data)-> np.ndarray:
   image= np.array(Image.open(BytesIO(data)))
   if image.shape[-1] == 4:
        image = image[..., :3] 
   return image

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    bytes = await file.read()
    image = read_file_as_image(bytes)
    img_batch = np.expand_dims(image,0)
    predictions =  MODEL.predict(img_batch)
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence= np.max(predictions[0])
    
   
    return {"message": f"{predictions} received and processed","class":predicted_class,"confidence":float(confidence)}

# Ensure the AugmentedImages directory exists
augmented_images_dir = "AugmentedImages"

if not os.path.exists(augmented_images_dir):
    os.makedirs(augmented_images_dir)


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip  = True,
    rotation_range = 10

)

train_generator = train_datagen.flow_from_directory(
    '../dataset/train',
    target_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = 32,
    class_mode = 'sparse',
    save_to_dir = "AugmentedImages"
)

for image_batch, label_batch in train_generator:
    print(image_batch.shape)
    break



if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1', port=8000)