import cv2


def start_mobilenet(ep_camera, device):
    prototxt = "MobileNetSSD_deploy.prototxt"
    caffe_model = "MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
    if device == 'cuda':
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ep_camera.start_video_stream(display=False, resolution="360p")
    window_closed = False
    classNames = {0: 'background',
                  1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                  5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                  10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                  14: 'motorbike', 15: 'person', 16: 'pottedplant',
                  17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

    while not window_closed:
        frame = ep_camera.read_cv2_image(strategy="newest")

        frame = frame.reshape(360, 640, 3)
        width = frame.shape[1]
        height = frame.shape[0]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 127.5, size=(300, 300), mean=(127.5, 127.5, 127.5),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()
        print(detections.shape[2])
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                x_top_left = int(detections[0, 0, i, 3] * width)
                y_top_left = int(detections[0, 0, i, 4] * height)
                x_bottom_right = int(detections[0, 0, i, 5] * width)
                y_bottom_right = int(detections[0, 0, i, 6] * height)

                cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                              (0, 255, 0))

                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    (w, h), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y_top_left = max(y_top_left, h)
                    cv2.rectangle(frame, (x_top_left, y_top_left - h),
                                  (x_top_left + w, y_top_left + t), (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x_top_left, y_top_left),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            window_closed = True
            ep_camera.stop_video_stream()
            break
        if cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) < 1:
            window_closed = True
            ep_camera.stop_video_stream()
            break
