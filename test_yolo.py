import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2

model = YOLO("yolov5s.pt")

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # -> si uso esto tengo que meter un .model mas

# Remove AutoShape
backbone = model.model.model.model          # DetectMultiBackend

# NOW this is the Sequential
print(type(backbone))     # should be nn.Sequential
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Extract backbone (layers 0–9)
backbone = nn.Sequential(*backbone[:10], nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
backbone.eval()

# Test
img = cv2.imread("group_photo_onefourth.png")
img = cv2.resize(img, (640, 640))

img = img.transpose(2, 0, 1)  # HWC to CHW


x = torch.randn(16, 3, 640, 640)
features = backbone(x)
print("Features shape: ", features.shape)

# results = model(imgs)
# print("Res: ", results)
