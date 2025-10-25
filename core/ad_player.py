import os, cv2

class AdPlayer:
    """Non-blocking player for video ads; call step() each loop."""
    def __init__(self, win="video"):
        self.win = win
        self._cap = None
        self._active = False

    def start(self, path):
        self.stop()
        if not path or not os.path.isfile(path):
            print(f"[!] Ad video not found: {path}")
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[!] Failed to open ad video: {path}")
            return
        self._cap = cap
        self._active = True
        try:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        except:
            pass

    def step(self):
        if not self._active or self._cap is None:
            return None
        ok, f = self._cap.read()
        if not ok or f is None:
            self.stop()
            return None
        cv2.imshow(self.win, f)
        return f

    def stop(self):
        if self._cap is not None:
            try: self._cap.release()
            except: pass
        self._cap = None
        self._active = False
        try:
            cv2.destroyWindow(self.win)
        except:
            pass

    def active(self):
        return self._active
