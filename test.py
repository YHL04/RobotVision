
from tensorflow.keras.models import load_model
import cv2
import numpy as np


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


class Predict:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        ret, frame = self.cam.read()
        self.frame_h, self.frame_w = frame.shape[0], frame.shape[1]

        self.model = load_model('model.h5')

        self.input_w, self.input_h = 416, 416
        self.anchors = [[116, 90, 156, 198, 373, 326],
                        [30, 61, 62, 45, 59, 119],
                        [10, 13, 16, 30, 33, 23]]
        self.class_threshold = 0.6
        self.overlap_thresh = 0.5
        self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                       "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                       "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                       "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                       "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                       "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                       "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                       "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                       "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        self.box_per_grid = 3
        self.num_classes = 80

        self.input_to_frame_h_scale = self.frame_h / self.input_h
        self.input_to_frame_w_scale = self.frame_w / self.input_w

    def decode_output(self, output):

        boxes = np.array([])

        for array in range(3):      # output contain list of three arrays
            out = output[array][0]
            num_column, num_row = out.shape[:2]
            out = out.reshape((num_column, num_row, self.box_per_grid, -1))

            out[..., :2] = _sigmoid(out[..., :2])
            out[..., 4:] = _sigmoid(out[..., 4:])
            out[..., 5:] = out[..., 4][..., np.newaxis] * out[..., 5:]
            out[..., 5:] *= out[..., 5:] > self.class_threshold

            grid_w = self.input_w / num_column
            grid_h = self.input_h / num_row

            for i in range(num_column*num_row):
                row = int(i / num_column)
                col = int(i % num_column)
                for b in range(self.box_per_grid):
                    objectness = out[row][col][b][4]
                    if objectness <= self.class_threshold: continue

                    x, y, w, h = out[row][col][b][:4]
                    x = (col + x) * grid_w  * self.input_to_frame_w_scale # center position
                    y = (row + y) * grid_h  * self.input_to_frame_h_scale # center position
                    w = self.anchors[array][2 * b + 0] * np.exp(w) * self.input_to_frame_w_scale
                    h = self.anchors[array][2 * b + 1] * np.exp(h) * self.input_to_frame_h_scale
                    classes = out[row][col][b][5:]

                    box = np.append(np.array([x-w/2, y-h/2, x+w/2, y+h/2]), classes)
                    boxes = np.append(boxes, box)

        boxes = np.reshape(boxes, (-1, 84))
        return boxes

    def non_max_suppression_fast(self, boxes):

        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > self.overlap_thresh)[0])))

        return boxes[pick]

    def main(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter('output.avi', fourcc, 12.0, (640, 480)) 
        # average around 80ms per frame
        while True:

            ret, frame = self.cam.read()
            raw = frame
            frame = cv2.resize(frame, (self.input_w, self.input_h))

            frame = frame.astype('float32')
            frame /= 255.0
            frame = np.expand_dims(frame,0)

            output = self.model.predict(frame)

            boxes = self.decode_output(output)
            boxes = self.non_max_suppression_fast(boxes)

            for i in range(len(boxes)):
                confidence = np.amax(boxes[i][4:])
                index = np.where(boxes[i][4:] == confidence)[0][0].astype('int')
                what_was_detected = self.labels[index]
                label = what_was_detected + ' ' + str(round(confidence * 100, 2))

                cv2.rectangle(raw, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])),
                              (255, 0, 0), 5)
                cv2.putText(raw, label, (int(boxes[i][0]), int(boxes[i][1]) + 30),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            print(boxes)
            cv2.imshow('object_detection', raw)

            out.write(raw) 
            
            
            if cv2.waitKey(1) == 27:
                break

        out.release()
        cv2.destroyAllWindows()
        

predict = Predict()
predict.main()


