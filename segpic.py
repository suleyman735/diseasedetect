import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tensorflow import keras
import tensorflow as tf
import cv2
import PIL
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers, models

# Load and preprocess your image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def generate_color_mask(image, lower_bound, upper_bound):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Create a mask for the specified color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Convert mask to float32 format
    mask = (mask > 0).astype(np.float32)
    return mask

# def clean_noise(mask, kernel_size=5):
#     # Create a kernel for morphological operations
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
#     # Perform morphological opening to remove small black dots
#     cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
#     return cleaned_mask

def apply_mask_to_image(image, mask):
    # Ensure the mask has the same number of channels as the image
    mask_3d = np.stack([mask] * 3, axis=-1)
    
    # Apply the mask to the image
    segmented_image = image * mask_3d
    
    # Convert the segmented image to uint8 format
    segmented_image = (segmented_image * 255).astype(np.uint8)
    return segmented_image

def process_image(image_path, output_path, lower_bound, upper_bound, kernel_size=5):
    try:
        # Load and preprocess the image
        img = Image.open(image_path).resize((256, 256))
        img_array = np.array(img) / 255.0  # Normalize to [0, 1] range

        # Convert image to RGB and back to uint8
        img_array_rgb = (img_array * 255).astype(np.uint8)

        # Generate mask
        green_mask = generate_color_mask(img_array_rgb, lower_bound, upper_bound)

        # Clean up small black dots in the mask
        # cleaned_mask = clean_noise(green_mask, kernel_size)

        # Apply the cleaned mask to the original image
        segmented_image = apply_mask_to_image(img_array, green_mask)

        # Save the segmented image
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(segmented_image).save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_directory(input_dir, output_dir, lower_bound, upper_bound, kernel_size=5):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Skip non-image files
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                process_image(input_path, output_path, lower_bound, upper_bound, kernel_size)

# Parameters
input_dir = 'images/'  # Directory containing the images
output_dir = 'processed_images/'  # Directory to save the segmented images

# Define the color range for green in HSV
lower_bound = np.array([0, 0, 0])  # Adjust these values as needed
upper_bound = np.array([115, 255, 255])

# Process all images in the input directory and save to the output directory
process_directory(input_dir, output_dir, lower_bound, upper_bound)

print("Image processing complete. Segmented images saved to:", output_dir)

# It is for one picture for 
# def generate_color_mask(image, lower_bound, upper_bound):
#     # Convert the image to HSV color space
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
#     # Create a mask for the specified color range
#     mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
#     # Convert mask to float32 format
#     mask = (mask > 0).astype(np.float32)
#     return mask

# def apply_mask_to_image(image, mask):
#     # Ensure the mask has the same number of channels as the image
#     mask_3d = np.stack([mask] * 3, axis=-1)
    
#     # Apply the mask to the image
#     segmented_image = image * mask_3d
    
#     # Convert the segmented image to uint8 format
#     segmented_image = (segmented_image * 255).astype(np.uint8)
#     return segmented_image




# # Load and preprocess the image
# image_path = 'IMA.JPG'
# img = Image.open(image_path).resize((256, 256))
# img_array = np.array(img) / 255.0  # Normalize to [0, 1] range

# # Convert image to RGB and back to uint8
# img_array_rgb = (img_array * 255).astype(np.uint8)

# # Define the color range for green in HSV
# lower_bound = np.array([0, 0, 0])  # Adjust these values as needed
# upper_bound = np.array([100, 255, 255])

# # Generate mask
# green_mask = generate_color_mask(img_array_rgb, lower_bound, upper_bound)



# # Apply the mask to the original image
# segmented_image = apply_mask_to_image(img_array, green_mask)

# # Display the original image with the segmented green areas
# plt.imshow(segmented_image)
# plt.title('Segmented Image with Green Areas')
# plt.axis('off')
# plt.show()
# //////////////////////////////
