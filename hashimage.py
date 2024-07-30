# this hash for imagehash start

from PIL import Image
import imagehash
import os

# Function to compute hash of an image
def compute_image_hash(image_path, hash_type):
    with Image.open(image_path) as img:
        if hash_type == 'phash':
            hash_value = imagehash.phash(img)
        elif hash_type == 'average':
            hash_value = imagehash.average_hash(img)
        elif hash_type == 'dhash':
            hash_value = imagehash.dhash(img)
        elif hash_type == 'whash':
            hash_value = imagehash.whash(img)
        elif hash_type == 'color':
            hash_value = imagehash.colorhash(img)
        else:
            raise ValueError("Unsupported hash type.")
    return hash_value


# Function to calculate similarity between two hashes
def compute_similarity(hash1, hash2):
    return hash1 - hash2
# Function to find the most similar image in the dataset using all hash types
# Function to find the most similar image in the dataset using all hash types

def is_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError, ValueError, Image.UnidentifiedImageError):
        return False


# Function to find the most similar image in the dataset using all hash types
def find_most_similar_image(query_image_path, dataset_folder):
    hash_types = ['phash', 'average', 'dhash', 'whash', 'color']
    query_hashes = {hash_type: compute_image_hash(query_image_path, hash_type) for hash_type in hash_types}
    
    min_similarity = float('inf')
    most_similar_image = None
    most_similar_dir = None

    for root, dirs, files in os.walk(dataset_folder):
        for image_name in files:
            image_path = os.path.join(root, image_name)
            if not is_image_file(image_path):
                continue
            
            total_similarity = 0
            
            for hash_type in hash_types:
                dataset_hash = compute_image_hash(image_path, hash_type)
                similarity = compute_similarity(query_hashes[hash_type], dataset_hash)
                total_similarity += similarity
            
            average_similarity = total_similarity / len(hash_types)
            
            if average_similarity < min_similarity:
                min_similarity = average_similarity
                most_similar_image = image_name
                most_similar_dir = os.path.basename(root)

    return most_similar_dir, most_similar_image, min_similarity

# Example usage
query_image_path = 'blackroot.jpg'
dataset_folder = 'images/'
most_similar_dir, most_similar_image, similarity = find_most_similar_image(query_image_path, dataset_folder)
print(f"Most similar image: {most_similar_image} in directory: {most_similar_dir} with average similarity score: {similarity}")

# end

# OpenCvHash start 
import os
import cv2
import numpy as np

# Function to compute hash of an image using OpenCV
def compute_image_hash(image_path, hash_type):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    
    if hash_type == 'phash':
        hash_value = cv2.img_hash.pHash(img)
    elif hash_type == 'average':
        hash_value = cv2.img_hash.AverageHash(img)
    elif hash_type == 'marr_hildreth':
        hash_value = cv2.img_hash.MarrHildrethHash_create().compute(img)
    elif hash_type == 'radial_variance':
        hash_value = cv2.img_hash.RadialVarianceHash_create().compute(img)
    elif hash_type == 'block_mean':
        hash_value = cv2.img_hash.BlockMeanHash_create(0)
    elif hash_type == 'color_moment':
        hash_value = cv2.img_hash.ColorMomentHash_create().compute(img)
    else:
        raise ValueError("Unsupported hash type.")
    
    return hash_value

# Function to calculate similarity between two hashes
def compute_similarity(hash1, hash2):
    return np.sum(hash1 != hash2)

# Function to check if a file is an image
def is_image_file(file_path):
    try:
        img = cv2.imread(file_path)
        if img is not None:
            return True
        else:
            return False
    except:
        return False

# Function to find the most similar image in the dataset using all hash types
def find_most_similar_image(query_image_path, dataset_folder):
    hash_types = ['phash', 'average', 'marr_hildreth', 'radial_variance', 'block_mean', 'color_moment']
    query_hashes = {hash_type: compute_image_hash(query_image_path, hash_type) for hash_type in hash_types}
    
    min_similarity = float('inf')
    most_similar_image = None
    most_similar_dir = None

    for root, dirs, files in os.walk(dataset_folder):
        for image_name in files:
            image_path = os.path.join(root, image_name)
            if not is_image_file(image_path):
                continue
            
            total_similarity = 0
            
            for hash_type in hash_types:
                dataset_hash = compute_image_hash(image_path, hash_type)
                similarity = compute_similarity(query_hashes[hash_type], dataset_hash)
                total_similarity += similarity
            
            average_similarity = total_similarity / len(hash_types)
            
            if average_similarity < min_similarity:
                min_similarity = average_similarity
                most_similar_image = image_name
                most_similar_dir = os.path.basename(root)

    return most_similar_dir, most_similar_image, min_similarity

# Example usage
query_image_path = 'blackroot.jpg'
dataset_folder = 'images/'

most_similar_dir, most_similar_image, similarity = find_most_similar_image(query_image_path, dataset_folder)
print(f"Most similar image: {most_similar_image} in directory: {most_similar_dir} with average similarity score: {similarity}")
# end


# with cnn model

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Function to extract features from an image using VGG16
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    vgg16_feature = model.predict(img_data)
    return vgg16_feature

# Function to check if a file is an image
def is_image_file(file_path):
    try:
        img = image.load_img(file_path)
        return True
    except:
        return False

# Function to find the most similar image in the dataset using CNN features
def find_most_similar_image(query_image_path, dataset_folder):
    query_features = extract_features(query_image_path, model)
    
    max_similarity = -1
    most_similar_image = None
    most_similar_dir = None

    for root, dirs, files in os.walk(dataset_folder):
        for image_name in files:
            image_path = os.path.join(root, image_name)
            if not is_image_file(image_path):
                continue
            
            dataset_features = extract_features(image_path, model)
            similarity = cosine_similarity(query_features, dataset_features)[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_image = image_name
                most_similar_dir = os.path.basename(root)

    return most_similar_dir, most_similar_image, max_similarity

# Example usage
query_image_path = 'blackroot.jpg'
dataset_folder = 'images/'

most_similar_dir, most_similar_image, similarity = find_most_similar_image(query_image_path, dataset_folder)
print(f"Most similar image: {most_similar_image} in directory: {most_similar_dir} with similarity score: {similarity}")



# most_similar_image, similarity = find_most_similar_image(query_image_path, dataset_folder)
# print(f"Most similar image: {most_similar_image} with average similarity score: {similarity}")

# Function to calculate similarity between two hashes
# def compute_similarity(hash1, hash2):
#     return hash1 - hash2
# hash_types = ['phash', 'average', 'dhash', 'whash', 'color']
# # Example usage
# image1_path = 'IMA.JPG'
# image2_path = 'IMA.JPG'

# Check and compare all hash types
# for hash_type in hash_types:
#     try:
#         hash1 = compute_image_hash(image1_path, hash_type)
#         hash2 = compute_image_hash(image2_path, hash_type)
        
#         similarity = compute_similarity(hash1, hash2)
#         print(f"Image similarity ({hash_type}): {similarity}")
#     except Exception as e:
#         print(f"Error with hash type {hash_type}: {e}")
# hash1 = compute_image_hash(image1_path, 'color')
# hash2 = compute_image_hash(image2_path, 'color')

# similarity = compute_similarity(hash1, hash2)
# print(f"Image similarity (phash): {similarity},Image similarity (average): {similarity},")
