import cv2

class PersonDetector:
    """Default to HOG-based pedestrian detector as a simple fallback.
    Replace with YOLO/ONNX/DFP for production.
    """
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def infer(self, frame):
        # returns [{'box':(x1,y1,x2,y2),'conf':1.0}, ...]
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8))
        out = []
        for (x,y,w,h), wgt in zip(rects, weights):
            out.append({'box': (x, y, x+w, y+h), 'conf': float(min(max(wgt, 0.0), 1.0))})
        return out
