
# first without segmentation
import os
from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from keras.layers import TFSMLayer
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import base64

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


# def read_file_as_image(data) -> np.ndarray:
#     image = Image.open(BytesIO(data))
#     image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Ensure image is resized to 256x256
#     image = np.array(image)
#     if image.shape[-1] == 4:
#         image = image[..., :3]
#     image = image / 255.0  # Normalize to [0, 1]
#     return image
def read_file_as_image(data)-> np.ndarray:
   image= np.array(Image.open(BytesIO(data)))
   
   if image.shape[-1] == 4:
        image = image[..., :3] 
   return image


# def calculate_ssim(image1, image2):
#     image1 = tf.image.convert_image_dtype(image1, tf.float32)
#     image2 = tf.image.convert_image_dtype(image2, tf.float32)
#     ssim = tf.image.ssim(image1, image2, max_val=1.0)
#     return tf.reduce_mean(ssim).numpy()

# def preprocess_dataset(image_paths, target_size=(256, 256)):
#     images = []
#     for image_path in image_paths:
#         image = Image.open(image_path).resize(target_size)
#         image = np.array(image) / 255.0
#         images.append(image)
#     return np.array(images)

def image_to_base64(image: np.ndarray) -> str:
    pil_image = Image.fromarray((image * 255).astype(np.uint8))  # Convert back to [0, 255] range
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# # Ensure that dataset_image_paths contains valid paths to images
# dataset_image_paths = [os.path.join(root, file) for root, _, files in os.walk('../dataset/train') for file in files if file.endswith(('JPG', 'jpeg', 'png'))]

# # Print to verify paths
# print(f"Found {len(dataset_image_paths)} images in dataset.")
# dataset_images = preprocess_dataset(dataset_image_paths)

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    bytes = await file.read()
    image = read_file_as_image(bytes)
    img_batch = np.expand_dims(image,0)
    predictions =  MODEL.predict(img_batch)
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence= np.max(predictions[0])
    

    
   
    return {"message": f"{predictions} received and processed","class":predicted_class,"confidence":float(confidence),
            # "similarity_percentage": similarity_percentage,"similar_image": 'similar_image_base64'
            }

# # Ensure the AugmentedImages directory exists
# augmented_images_dir = "AugmentedImages"

# if not os.path.exists(augmented_images_dir):
#     os.makedirs(augmented_images_dir)


# train_datagen = ImageDataGenerator(
#     rescale = 1./255,
#     horizontal_flip  = True,
#     rotation_range = 10

# )
# # print(train_datagen)

# train_generator = train_datagen.flow_from_directory(
#     '../dataset/train',
#     target_size = (IMAGE_SIZE,IMAGE_SIZE),
#     batch_size = 32,
#     class_mode = 'sparse',
#     save_to_dir = "AugmentedImages"
# )

# for image_batch, label_batch in train_generator:
#     # print(image_batch , label_batch)
#     break



if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1', port=8000)
    
    
    
    
#             # Calculate similarity with dataset images
#     # max_similarity = 0
#     # most_similar_image = None
#     # for data_image in dataset_images:
#     #     similarity = calculate_ssim(image, data_image)
#     #     if similarity > max_similarity:
#     #         max_similarity = similarity
#     #         most_similar_image = data_image
    
    
#     # similarity_percentage = max_similarity * 100
#     # similar_image_base64 = image_to_base64(most_similar_image) if most_similar_image is not None else None
    
#         # Visualization for debug purposes (not necessary in production)
#     # if most_similar_image is not None:
#     #     pass
#         # plt.figure(figsize=(10, 5))
#         # plt.subplot(1, 2, 1)
#         # plt.title('Uploaded Image')
#         # plt.imshow(image)
#         # plt.subplot(1, 2, 2)
#         # plt.title(f'Similar Image (Similarity: {similarity_percentage:.2f}%)')
#         # plt.imshow(most_similar_image)
#         # plt.show()






# it is with segmantation both
# import os
# from fastapi import FastAPI, File, UploadFile
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import cv2

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import base64

# IMAGE_SIZE = 256
# CHANNELS = 3

# app = FastAPI()
# MODEL_PATH = "../models_processed_images/3/model.keras"
# MODEL = tf.keras.models.load_model(MODEL_PATH)
# CLASS_NAMES = ["Apple scab", "Black Rot", "Cedar Apple Rust", "Healthy"]

# @app.get("/ping")
# async def ping():
#     return "Hello, World"

# def read_file_as_image(data)-> np.ndarray:
#    image= np.array(Image.open(BytesIO(data)))
   
#    if image.shape[-1] == 4:
#         image = image[..., :3] 
#    return image


# def calculate_ssim(image1, image2):
#     image1 = tf.image.convert_image_dtype(image1, tf.float32)
#     image2 = tf.image.convert_image_dtype(image2, tf.float32)
#     ssim = tf.image.ssim(image1, image2, max_val=1.0)
#     return tf.reduce_mean(ssim).numpy()

# def preprocess_dataset(image_paths, target_size=(256, 256)):
#     images = []
#     for image_path in image_paths:
#         image = Image.open(image_path).resize(target_size)
#         image = np.array(image) / 255.0
#         images.append(image)
#     return np.array(images)

# def image_to_base64(image: np.ndarray) -> str:
#     pil_image = Image.fromarray((image * 255).astype(np.uint8))
#     buffered = BytesIO()
#     pil_image.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return img_str

# def generate_color_mask(image, lower_bound, upper_bound):
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
#     mask = (mask > 0).astype(np.float32)
#     return mask

# def apply_mask_to_image(image, mask):
#     mask_3d = np.stack([mask] * 3, axis=-1)
#     segmented_image = image * mask_3d
#     segmented_image = (segmented_image * 255).astype(np.uint8)
#     return segmented_image

# # dataset_image_paths = [os.path.join(root, file) for root, _, files in os.walk('../dataset/train') for file in files if file.endswith(('JPG', 'jpeg', 'png'))]
# # print(f"Found {len(dataset_image_paths)} images in dataset.")
# # dataset_images = preprocess_dataset(dataset_image_paths)

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     bytes = await file.read()
#     image = read_file_as_image(bytes)
    
#     # Define the color range for segmentation (example values, you may need to adjust these)
#     lower_bound = (0, 0, 0)  # Lower bound of HSV for green color
#     upper_bound = (115, 255, 255)  # Upper bound of HSV for green color
    
#     # Convert image to RGB and uint8 format
#     img_array_rgb = (image * 255).astype(np.uint8)
    
#     # Generate mask and apply it to the image
#     green_mask = generate_color_mask(img_array_rgb, lower_bound, upper_bound)
#     segmented_image = apply_mask_to_image(image, green_mask)
    
#     # Preprocess the segmented image for the model
#     segmented_image = segmented_image / 255.0
#     img_batch = np.expand_dims(segmented_image, 0)
    
#     predictions = MODEL.predict(img_batch)
#     index = np.argmax(predictions[0])
#     predicted_class = CLASS_NAMES[index]
#     confidence = np.max(predictions[0])
    
#     return {
#         "message": f"{predictions} received and processed",
#         "class": predicted_class,
#         "confidence": float(confidence)
#     }

# augmented_images_dir = "AugmentedImages"
# if not os.path.exists(augmented_images_dir):
#     os.makedirs(augmented_images_dir)

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     rotation_range=10
# )

# train_generator = train_datagen.flow_from_directory(
#     '../dataset/train',
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=32,
#     class_mode='sparse',
#     save_to_dir="AugmentedImages"
# )

# # for image_batch, label_batch in train_generator:
# #     break

# if __name__ == "__main__":
#     uvicorn.run(app, host='127.0.0.1', port=8000)