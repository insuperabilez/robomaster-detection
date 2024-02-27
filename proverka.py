from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import numpy as np
from robomaster import robot
import cv2
"""
image = Image.open('image.jpg')
img = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
print(image.size) #(640, 425)
print(img.size) #816000
print([image.size[::-1]]) #[(425, 640)]
print('-----')

#print(image.shape) 'JpegImageFile' object has no attribute 'shape'
print(img.shape) #(425, 640, 3)
ep_robot = robot.Robot()
ep_robot.initialize()
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False)

img = ep_camera.read_cv2_image()
image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
print(image.size)
print(img.shape)
print([image.size[::-1]])
"""
from ultralytics import YOLO
model = YOLO("yolo-Weights/yolov8n.pt")
print(model.names)