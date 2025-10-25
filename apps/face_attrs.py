import os, cv2, numpy as np
try:
    import dlib
    _HAS_DLIB = True
except Exception:
    _HAS_DLIB = False

class FaceAttrEstimator:
    def __init__(self):
        self.gender_proto = os.getenv('GENDER_PROTOTXT', 'models/gender_deploy.prototxt')
        self.gender_model = os.getenv('GENDER_MODEL', 'models/gender_net.caffemodel')
        self.age_proto    = os.getenv('AGE_PROTOTXT', 'models/age_deploy.prototxt')
        self.age_model    = os.getenv('AGE_MODEL', 'models/age_net.caffemodel')
        self.gender_net = self._try_load(self.gender_proto, self.gender_model)
        self.age_net    = self._try_load(self.age_proto, self.age_model)
        self.face_det = dlib.get_frontal_face_detector() if _HAS_DLIB else None

    def _try_load(self, proto, model):
        try:
            if os.path.exists(proto) and os.path.exists(model):
                return cv2.dnn.readNetFromCaffe(proto, model)
        except Exception:
            pass
        return None

    def _faces(self, frame):
        if self.face_det is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.face_det(gray, 0)
        out = []
        for r in rects:
            x, y, w, h = r.left(), r.top(), r.width(), r.height()
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
            if x2 > x1 and y2 > y1:
                out.append((x1,y1,x2,y2))
        return out

    def infer(self, frame):
        results = []
        for (x1,y1,x2,y2) in self._faces(frame):
            face = frame[y1:y2, x1:x2]
            age, gender = None, None
            if self.gender_net is not None and self.age_net is not None:
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                             (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                self.gender_net.setInput(blob)
                g = self.gender_net.forward()
                gender = 'Male' if g[0][0] > g[0][1] else 'Female'
                self.age_net.setInput(blob)
                a = self.age_net.forward()
                age_index = int(np.argmax(a))
                age = int((age_index + 1) * 3.8)
            # emotion placeholder (requires DeepFace if desired)
            results.append({'box': (x1,y1,x2,y2), 'age': age, 'gender': gender, 'emotion': None})
        return results
