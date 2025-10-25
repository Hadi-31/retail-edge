
import os, cv2, argparse, time, numpy as np, dlib
from apps.person_detect import PersonDetector
from core.tracking import IOUTracker
from core.heatmap_simple import SimpleHeatmap
from core.ad_engine import AdEngine

# ===== Config (simple) =====
FSKIP        = int(os.getenv("FRAME_SKIP", "0"))
MIN_SCORE    = float(os.getenv("MIN_SCORE", "0.30"))
HEAT_OUT_DIR = os.getenv("HEAT_OUT_DIR", "heatmap_reports")
WINDOW_SEC   = float(os.getenv("WINDOW_SEC", "3.0"))
GENDER_CONF  = float(os.getenv("GENDER_CONF_THR", "0.90"))
EMA_ALPHA    = float(os.getenv("EMA_ALPHA", "0.6"))

# Caffe age/gender
GENDER_PROTOTXT  = os.getenv("GENDER_PROTOTXT", "models/gender_deploy.prototxt")
GENDER_MODEL     = os.getenv("GENDER_MODEL",    "models/gender_net.caffemodel")
AGE_PROTOTXT     = os.getenv("AGE_PROTOTXT",    "models/age_deploy.prototxt")
AGE_MODEL        = os.getenv("AGE_MODEL",       "models/age_net.caffemodel")

AGE_MIDS = np.array([1, 5, 10, 18, 28, 40.5, 50.5, 65], dtype=np.float32)

def softmax(x):
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def expected_age(logits_or_probs):
    v = np.array(logits_or_probs, dtype=np.float32).flatten()
    p = v / (np.sum(v) + 1e-8)
    if not (0.99 <= p.sum() <= 1.01):
        p = softmax(v)
    return float(np.dot(p, AGE_MIDS))

class AdPlayer:
    def __init__(self, win="video"):
        self.win = win
        self.cap = None
        self.img = None
        self.active = False
        self.is_image = False

    def start(self, path):
        self.stop()
        if not path or not os.path.exists(path):
            return False
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        if ok and frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap = cap
            self.img = None
            self.is_image = False
            self.active = True
            return True
        cap.release()
        img = cv2.imread(path)
        if img is not None:
            self.img = img
            self.is_image = True
            self.active = True
            return True
        return False

    def step(self):
        if not self.active:
            return False
        if self.is_image:
            cv2.imshow(self.win, self.img)
            return True
        ok, f = self.cap.read()
        if not ok or f is None:
            self.stop(); return False
        cv2.imshow(self.win, f)
        return True

    def stop(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        self.cap = None
        self.img = None
        self.active = False
        self.is_image = False
        try: cv2.destroyWindow(self.win)
        except: pass

def parse_args():
    ap = argparse.ArgumentParser("retail-edge (simple): detection + nonblocking ads + simple heatmap")
    ap.add_argument("--source", type=str, default="0", help="webcam index or video path")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--videos-dir", type=str, default="ui/assets/ads")
    ap.add_argument("--kids-age", type=int, default=12)
    return ap.parse_args()

def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    cam_id = str(args.source)

    # Load DNNs (if present)
    gender_net = None; age_net = None
    try:
        if os.path.exists(GENDER_PROTOTXT) and os.path.exists(GENDER_MODEL):
            gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT, GENDER_MODEL)
        if os.path.exists(AGE_PROTOTXT) and os.path.exists(AGE_MODEL):
            age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_MODEL)
    except Exception as e:
        print("[simple] Failed loading age/gender models:", e)
        gender_net, age_net = None, None

    face_det = dlib.get_frontal_face_detector()
    det = PersonDetector()
    trk = IOUTracker(iou_thresh=0.4, max_age=30)
    adeng = AdEngine(kids_age=args.kids_age)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[simple] Could not open source:", args.source); return
    ok, f0 = cap.read()
    if not ok or f0 is None:
        print("[simple] No frames."); return

    heat = SimpleHeatmap(f0.shape, grid=64)
    try: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except: pass

    totals = {"faces":0, "age_sum":0.0, "male":0, "female":0}
    age_ema = None
    window_ts = time.time()

    frame_i = 0
    cam_win = "camera"
    ad = AdPlayer()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None: break

        if FSKIP and (frame_i % (FSKIP + 1) != 0):
            frame_i += 1
            if not args.no_display:
                cv2.imshow(cam_win, frame)
                if ad.active: ad.step()
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')): break
                if k in (ord('c'),): ad.stop()
            continue

        dets = det.infer(frame) or []
        dets = [d for d in dets if d.get('conf', 1.0) >= MIN_SCORE]
        tracks = trk.update(dets)

        heat.update(tracks)

        # faces + age/gender
        H,W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_det(gray, 0)
        for r in rects:
            x,y,w,h = r.left(), r.top(), r.width(), r.height()
            x = max(0,x); y = max(0,y); w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
            sub = frame[y:y+h, x:x+w]
            if sub.size == 0: continue
            age_ev, g_lbl, g_conf = None, None, None
            if gender_net is not None and age_net is not None:
                blob = cv2.dnn.blobFromImage(sub, 1.0, (227,227), (78.4263377603,87.7689143744,114.895847746), swapRB=False)
                gender_net.setInput(blob)
                g = gender_net.forward()[0]
                gp = softmax(g)
                if gp[0] >= gp[1]:
                    g_lbl, g_conf = "Male", float(gp[0])
                else:
                    g_lbl, g_conf = "Female", float(gp[1])

                age_net.setInput(blob)
                a = age_net.forward()[0]
                age_ev = expected_age(a)

            if g_conf is not None and g_conf < GENDER_CONF:
                g_lbl = "unsure"

            if age_ev is not None:
                age_ema = age_ev if age_ema is None else (EMA_ALPHA*age_ev + (1-EMA_ALPHA)*age_ema)

            if g_lbl == "Male": totals["male"] += 1
            elif g_lbl == "Female": totals["female"] += 1
            if age_ev is not None: totals["age_sum"] += float(age_ev)
            totals["faces"] += 1

            # simple overlay
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            label = f"{g_lbl or '?'} {int(round(age_ev)) if age_ev is not None else ''}"
            cv2.putText(frame, label, (x, max(8,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # decision window
        if (not ad.active) and (time.time() - window_ts >= WINDOW_SEC):
            if totals["faces"] > 0:
                avg_age = totals["age_sum"] / totals["faces"]
                frac_male = totals["male"] / max(1, totals["faces"])
                ad_path = adeng.choose(avg_age, frac_male, base=args.videos_dir)
                if ad_path: ad.start(ad_path)
                print(f"[simple] Decision: avg_age={avg_age:.1f} male%={frac_male*100:.1f} -> {ad_path}")
            totals = {"faces":0, "age_sum":0.0, "male":0, "female":0}
            window_ts = time.time()

        # render heatmap overlay
        vis = heat.render(frame)

        if age_ema is not None:
            cv2.putText(vis, f"Age(EMA): {age_ema:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,255), 2, cv2.LINE_AA)

        if not args.no_display:
            cv2.imshow(cam_win, vis)
            if ad.active: ad.step()
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')): break
            if k in (ord('c'),): ad.stop()

        frame_i += 1

    # save outputs
    try:
        j, p = heat.save(HEAT_OUT_DIR, cam_id)
        print("[simple] Saved JSON:", j)
        print("[simple] Saved PNG :", p)
    except Exception as e:
        print("[simple] Failed to save heatmap:", e)

    ad.stop()
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
