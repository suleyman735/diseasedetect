import cv2


# loading mask rcnn
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


img = cv2.imread("road.jpg")
blob = cv2.dnn.blobFromImage(img,swapRB=True)
net.setInput(blob)

boxes,masks= net.forward(["detection_out_final","detection_masks"])
box = boxes[0,0,1]

print(boxes)
cv2.imshow("Image",img)
cv2.waitKey(0)