from PIL import Image
import imagehash

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

# Example usage
image1_path = 'IMA.JPG'
image2_path = 'ROAD.JPG'

hash1 = compute_image_hash(image1_path, 'color')
hash2 = compute_image_hash(image2_path, 'color')

similarity = compute_similarity(hash1, hash2)
print(f"Image similarity (phash): {similarity},Image similarity (average): {similarity},")
