from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
image = Image.open('image.jpg')
def start_segment():
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    img_float32 = np.float32(pred_seg*15)
    #img = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR)
    cv2.imshow("Segmentation", img_float32)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
