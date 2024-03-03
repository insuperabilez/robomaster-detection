from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import numpy as np
from robomaster import robot
import cv2
#image = Image.open('image.jpg')
#img = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
def start_resnet(ep_camera):
    print('start')
    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
    ep_camera.start_video_stream(display=False)
    #ep_camera.start_video_stream(display=False,resolution=camera.STREAM_360P)
    for i in range(100):
        img = ep_camera.read_cv2_image()
        #img = ep_camera.read_cv2_image(strategy="newest")
        #img = Image.fromarray(img)
        #img= img.reshape(720, 1280,3) работает
        image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        print(image.size)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        #target_sizes = torch.tensor([image.size[::-1]])


        print(len(outputs))

        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
            )
        boxes = [box.tolist() for box in results["boxes"]]
        for box, label in zip(boxes, results["labels"]):
            x1, y1, x2, y2 = box
            class_name = model.config.id2label[label.item()]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
