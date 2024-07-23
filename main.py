import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# print('hotel')
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
dataset = tf.keras.preprocessing.image_dataset_from_directory('images',
                                                    shuffle = True,
                                                    image_size = (IMAGE_SIZE,IMAGE_SIZE),
                                                    batch_size = BATCH_SIZE)


class_name = dataset.class_names

# for image_batch,label_batch in dataset.take(1):
#     for i in range(12):
#         ax = plt.subplot(3,4,i+1)
#         print(ax)
#         plt.imshow(image_batch[0].numpy().astype("uint8"))
#         plt.title(class_name[label_batch[0]])
#         print(plt.title(class_name[label_batch[0]]))
#         plt.axis("off")
    # print(image_batch[0].shape)
    # print(label_batch.numpy())
    
def get_dataset_partitions_tf(ds,train_split = 0.8,val_split = 0.1,test_split = 0.1,shuffle =True,shuffle_size = 10000):
    ds_size=len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size,seed=12)
        
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    
    return  train_ds, val_ds, test_ds
    
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
# print(len(train_ds))
# print(len(val_ds))
# print(len(test_ds))

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
  layers.RandomRotation(0.2)
])

data_augmentation = tf.keras.Sequential([
   layers.RandomFlip("horizontal_and_vertical"),
   layers.RandomRotation(0.2),
])
input_shape = (IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes = 4
model = models.Sequential([
    layers.InputLayer(shape=input_shape),
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu',),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu',),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu',),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu',),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu',),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax',)
])
model.summary()
# model.build(input_shape=input_shape)
# print(model)

model.compile(
    optimizer="adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)

history = model.fit(
    train_ds,
    epochs =EPOCHS,
    batch_size = BATCH_SIZE,
    verbose = 1,
    validation_data = val_ds
)
# print(history.history)

plt.figure(figsize=(15,15))
def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy(),)
    img_array = tf.expand_dims(img_array,0)
    
    predictions = model.predict(img_array)
    
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class , confidence

for images,labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class,confidence = predict(model,images[i].numpy())
        actual_class = class_name[labels[i]]
        plt.title(f"Actuakl: {actual_class}, \n Prediction: {predicted_class}.\n Confidence: {confidence}")
        
        plt.axis("off")



model_version = max([int(i) for i in os.listdir("models/") if i.isdigit()] + [0]) + 1
model_dir = f"models/{model_version}"

# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Save the model in the desired format
model.save(os.path.join(model_dir, "model.keras"))

# model_version = 1
# model_dir = f"models/{model_version}"
# model.save(f"models/{model_version}/model.h5")


# model_version = max([int(i) for i in os.listdir("models/")+[0]])+1
# model.save(f"/models/{model_version}/")
# Visualize training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# img = cv2.imread("road.jpg")

# cv2.imshow('image',img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()




# def read_images_from_folder(folder_paths):
#     image_data = []
#     for folder_path in folder_paths:
#         for filename in os.listdir(folder_path):
#             if filename.endswith((".JPG", ".jpeg", ".png", ".bmp", ".tiff", ".gif")):
#                 img_path = os.path.join(folder_path, filename)
#                 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#                 if img is None:
#                     print(f"Error: Could not read image {img_path}")
#                 else:
#                     image_data.append((img, os.path.basename(folder_path), filename))
#     return image_data

# def image_in_folders(specific_image, image_data):
#     if specific_image is None:
#         print("Error: The specific image is empty or could not be read.")
#         return None

#     sift = cv2.SIFT_create()
#     print(sift)
#     kp1, des1 = sift.detectAndCompute(specific_image, None)
#     # print(des1)

#     if des1 is None:
#         print("Error: No descriptors found in the specific image.")
#         return None

#     for img, folder_name, image_name in image_data:
#         kp2, des2 = sift.detectAndCompute(img, None)

#         if des2 is None:
#             continue

#         # Using BFMatcher to find the best matches
#         bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#         matches = bf.match(des1, des2)

#         # If enough matches are found, consider the images similar
#         if len(matches) > 0.1 * len(kp1):  # Adjust threshold as necessary
#             return folder_name, image_name
#     return None

# # Example usage:
# folder_path = ['images/Apple___Apple_scab','images/Apple___Black_rot','images/Apple___Cedar_apple_rust','images/Apple___healthy']
# image_data = read_images_from_folder(folder_path)

# # Print out the number of images read
# print(f"Number of images read: {len(image_data)}")

# # Check if a specific image is in the list
# specific_image_path = 'road.JPG'
# specific_image = cv2.imread(specific_image_path, cv2.IMREAD_COLOR)



# if specific_image is None:
#     print(f"Error: Could not read specific image {specific_image_path}")
# else:
#     result = image_in_folders(specific_image, image_data)
#     print(result)
#     if result:
#         print(f"The specific image is found in directory '{result[0]}' with the name '{result[1]}'.")
#     else:
#         print("The specific image is not in the list.")


    
# cv2.imshow("difference", diff)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# loading mask rcnn
# net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


# img = cv2.imread("road1.jpg")
# blob = cv2.dnn.blobFromImage(img,swapRB=True)
# net.setInput(blob)

# boxes,masks= net.forward(["detection_out_final","detection_masks"])
# box = boxes[0,0,1]


# cv2.imshow("Image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# mask 
# net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# img = cv2.imread('road1.jpg')
# height, width, _ = img.shape


# # Black image
# # black_image = np.zeros((height,width,1),np.uint8)

# blob = cv2.dnn.blobFromImage(img,swapRB=True)
# net.setInput(blob)
# boxes,masks = net.forward(["detection_out_final","detection_masks"])
# detection_count = boxes.shape[2]
# for i in range(detection_count):

#     box = boxes[0,0,i]
#     # print(box)
#     class_id = box[1]
#     print(box)

# cv2.imshow("Detected Objects with Polygons and Coordinates", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()