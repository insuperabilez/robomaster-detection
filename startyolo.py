from ultralytics import YOLO
import cv2
import math
#image = cv2.imread("image.jpg")
def start_yolo(ep_camera):

    model = YOLO("yolo-Weights/yolov8n.pt")

    # Настраиваем видеопоток с веб-камеры
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    window_closed = False
    ep_camera.start_video_stream(display=False,resolution="360p")
    while not window_closed:
        #img=image.copy()
        img = ep_camera.read_cv2_image()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])


                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (0, 255, 0)
                thickness = 2


                cv2.putText(img, model.names[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Detection', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            window_closed = True
            ep_camera.stop_video_stream()
            break

            # Проверка закрытия окна
        if cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) < 1:
            window_closed = True
            ep_camera.stop_video_stream()
            break
