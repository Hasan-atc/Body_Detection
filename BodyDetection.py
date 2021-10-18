import cv2
import numpy as np

vid = cv2.VideoCapture("Video PATH or webcam(0)")

while True:
    x, frame = vid.read()

    en = frame.shape[1]
    boy = frame.shape[0]
    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

    layers = model.getLayerNames()
    output_layers = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)
    detect_layers = model.forward(output_layers)

    ids_list = []
    boxes_list = []
    confidence_list = []

    for detect_layer in detect_layers:
        for o_detection in detect_layer:

            skor = o_detection[5:]
            p_id = np.argmax(skor)
            confidence = skor[p_id]

            if confidence > 0.5:
                label = classes[p_id]
                box = o_detection[0:4] * np.array([en, boy, en, boy])
                (box_center_x, box_center_y, box_en, box_boy) = box.astype("int")

                start_x = int(box_center_x - (box_en/2))
                start_y = int(box_center_y - (box_en/2))

                ids_list.append(p_id)
                confidence_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_en), int(box_boy)])

                max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)

                for max_id in max_ids:
                    max_class_id = max_id[0]
                    box = boxes_list[max_class_id]

                    start_x = box[0]
                    start_y = box[1]
                    box_en = box[2]
                    box_boy = box[3]

                    p_id = ids_list[max_class_id]
                    label = classes[p_id]
                    confidence = confidence_list[max_class_id]

                end_x = start_x + box_en
                end_y = start_y + box_boy

                b_color = colors[p_id]
                b_color = [int(people) for people in b_color]

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), b_color, 2)
                cv2.putText(frame, label, (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()