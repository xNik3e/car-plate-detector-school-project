import cv2
import numpy as np
import os

filename = '003.mp4'

class LicensePlateDetector:
    def __init__(self, pth_weights: str, pth_cfg: str, pth_classes: str):
        self.net = cv2.dnn.readNet(pth_weights, pth_cfg)
        self.classes = []
        with open(pth_classes, 'r') as f:
            self.classes = f.read().splitlines()
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 0, 0)
        self.coordinates = None
        self.img = None
        self.fig_image = None
        self.roi_image = None
        self.current_frame = 0
        
        
    def detect(self, image):
        orig = image
        self.img = orig
        img = orig.copy()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layer_names)
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores) 
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                self.roi_image = img[y:y + h, x:x + w]
                try:
                    name = './data/licence_plate' + str(self.current_frame) + '.jpg'
                    self.current_frame += 1
                    cv2.imwrite(name, self.roi_image)
                except:
                    pass
                cv2.rectangle(img, (x,y), (x + w, y + h), self.color, 5)
                
        self.fig_image = img
        self.coordinates = (x, y, w, h)
        return
    
# Playing video from file:
cap = cv2.VideoCapture(filename)
lpd = LicensePlateDetector(
    pth_weights='yolov3_training_final.weights', 
    pth_cfg='yolov3_testing.cfg', 
    pth_classes='classes.txt'
)

def display_frame(frame, delay):
    
    try:
        lpd.detect(frame)
        cv2.imshow('frame', lpd.fig_image)
        
        cv2.waitKey(delay)    
    except UnboundLocalError:
        cv2.imshow('frame', frame)
        cv2.waitKey(delay)


try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

FPS_IN = cap.get(cv2.CAP_PROP_FPS) 
FPS_OUT = FPS_IN // 5

delay = int((1000/ FPS_OUT))

index_in = -1
index_out = -1
currentFrame = 0
while(True):
    # Capture frame-by-frame
    success = cap.grab()
    if not success: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        lpd.current_frame = 0
    index_in += 1
    
    out_due = int(index_in/ FPS_IN * FPS_OUT)
    if out_due > index_out:
        success, frame = cap.retrieve()
        if not success: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        index_out += 1
        display_frame(frame, delay)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

