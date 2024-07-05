import os
import cv2
import numpy as np

print('jeelo')

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
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

img = cv2.imread('road1.jpg')
height, width, _ = img.shape


# Black image
# black_image = np.zeros((height,width,1),np.uint8)

blob = cv2.dnn.blobFromImage(img,swapRB=True)
net.setInput(blob)
boxes,masks = net.forward(["detection_out_final","detection_masks"])
detection_count = boxes.shape[2]
for i in range(detection_count):

    box = boxes[0,0,i]
    # print(box)
    class_id = box[1]
    print(box)

cv2.imshow("Detected Objects with Polygons and Coordinates", img)
cv2.waitKey(0)
cv2.destroyAllWindows()