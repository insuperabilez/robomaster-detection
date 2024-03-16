import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection


# image = Image.open('image.jpg')
# img = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
def start_resnet101(ep_camera, device):
    window_closed = False
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
    model = model.to(device)
    ep_camera.start_video_stream(display=False, resolution="360p")
    # ep_camera.start_video_stream(display=False,resolution=camera.STREAM_360P)
    while not window_closed:
        img = ep_camera.read_cv2_image(strategy="newest")
        # img = Image.fromarray(img)
        img = img.reshape(360, 640, 3)
        # image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        inputs = processor(images=img, return_tensors="pt")
        inputs.to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([img.shape[0:2]]).to(device)
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

        cv2.imshow("Detection", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            window_closed = True
            ep_camera.stop_video_stream()
            break
        if cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) < 1:
            window_closed = True
            ep_camera.stop_video_stream()
            break
