import os, cv2, argparse, time, numpy as np
import dlib
from apps.person_detect import PersonDetector
from core.tracking import IOUTracker
from core.heatmap_tracker import HeatmapTracker
from core.fusion import fuse
from core.ad_engine import AdEngine

# ========= Environment / Config =========
FSKIP            = int(os.getenv("FRAME_SKIP", "0"))
MIN_SCORE        = float(os.getenv("MIN_SCORE", "0.30"))
DWELL_THRESH     = float(os.getenv("DWELL_THRESH", "5"))
HOT_THRESH       = float(os.getenv("HOT_THRESH", "10"))
HEAT_OUT_DIR     = os.getenv("HEAT_OUT_DIR", "heatmap_reports")
DRAW_BOXES       = os.getenv("DRAW_BOXES", "1") == "1"

# OpenCV DNN age/gender model paths (Caffe)
GENDER_PROTOTXT  = os.getenv("GENDER_PROTOTXT", "models/gender_deploy.prototxt")
GENDER_CAFFE     = os.getenv("GENDER_MODEL",    "models/gender_net.caffemodel")
AGE_PROTOTXT     = os.getenv("AGE_PROTOTXT",    "models/age_deploy.prototxt")
AGE_CAFFE        = os.getenv("AGE_MODEL",       "models/age_net.caffemodel")

# Decision window & thresholds
WINDOW_SEC       = float(os.getenv("WINDOW_SEC", "3.0"))      # seconds between demographic decisions
EMA_ALPHA        = float(os.getenv("EMA_ALPHA", "0.6"))       # age smoothing
GENDER_CONF_THR  = float(os.getenv("GENDER_CONF_THR", "0.90"))

# Age-bin midpoints used for expected-value age (tune to your model bins)
AGE_MIDS = np.array([1, 5, 10, 18, 28, 40.5, 50.5, 65], dtype=np.float32)


# ========= Helpers =========
def must_exist(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[retail-edge] Missing {label}: {path}")

def softmax(x):
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def age_expected_value(logits_or_probs, mids=AGE_MIDS):
    """
    Accepts raw logits OR probabilities. If they don't sum ~1, we softmax.
    Returns expected age = sum(p_i * mid_i).
    """
    vec = np.array(logits_or_probs, dtype=np.float32).flatten()
    p = vec / (np.sum(vec) + 1e-8)
    if not (0.99 <= p.sum() <= 1.01):
        p = softmax(vec)
    return float(np.dot(p, mids))

def clip_roi(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


class AdPlayer:
    """Non-blocking video/image player. Call step() each loop; shows frames without blocking capture."""
    def __init__(self, win_name="video"):
        self.win = win_name
        self.cap = None
        self.is_image = False
        self.stopped = True
        self.static_frame = None

    def start(self, path):
        self.stop()
        if not path or not os.path.exists(path):
            print(f"[retail-edge] Ad not found: {path}")
            return False

        # Try open as video
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        if ok and frame is not None:
            # It is a video; keep cap open and rewind
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap = cap
            self.is_image = False
            self.stopped = False
            print(f"[retail-edge] Playing ad (video): {path}")
            return True

        # Fallback as image
        cap.release()
        img = cv2.imread(path)
        if img is not None:
            self.static_frame = img
            self.is_image = True
            self.stopped = False
            print(f"[retail-edge] Showing ad (image): {path}")
            return True

        print(f"[retail-edge] Failed to play ad: {path}")
        return False

    def step(self):
        """Advance and display one frame. Returns False if finished or stopped."""
        if self.stopped:
            return False

        if self.is_image:
            cv2.imshow(self.win, self.static_frame)
            # keep image up; don't auto-stop
            return True

        # video mode
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.stop()
            return False
        cv2.imshow(self.win, frame)
        return True

    def stop(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.cap = None
        self.static_frame = None
        self.stopped = True
        self.is_image = False
        try:
            cv2.destroyWindow(self.win)
        except:
            pass

    def active(self):
        return not self.stopped


def parse_args():
    ap = argparse.ArgumentParser("retail-edge: tracking + demographics + ads + heatmap")
    ap.add_argument("--source", type=str, default="0", help="webcam index or video file")
    ap.add_argument("--no-display", action="store_true", help="run headless")
    ap.add_argument("--videos-dir", type=str, default="ui/assets/ads", help="ad assets directory")
    # Keep --interval/--kids-age for backward compat, but WINDOW_SEC/personas.yaml drive behavior now
    ap.add_argument("--interval", type=float, default=WINDOW_SEC, help="(deprecated) seconds between decisions")
    ap.add_argument("--kids-age", type=int, default=12, help="(deprecated) threshold; personas.yaml is preferred")
    return ap.parse_args()


def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    cam_id = str(args.source)

    # Validate DNN model files (we run age/gender here to access logits)
    try:
        must_exist(GENDER_PROTOTXT, "GENDER_PROTOTXT")
        must_exist(GENDER_CAFFE,   "GENDER_MODEL")
        must_exist(AGE_PROTOTXT,   "AGE_PROTOTXT")
        must_exist(AGE_CAFFE,      "AGE_MODEL")
    except FileNotFoundError as e:
        print(e)
        print("[retail-edge] Continuing without age/gender (overlays still run, ads may default).")
        # We will set nets to None below.

    # Load nets (if available)
    gender_net = None
    age_net = None
    try:
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT, GENDER_CAFFE)
        age_net    = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT,    AGE_CAFFE)
    except Exception as e:
        print(f"[retail-edge] Failed to load age/gender nets: {e}")
        gender_net, age_net = None, None

    # Dlib face detector (landmarks not required)
    face_det = dlib.get_frontal_face_detector()

    det  = PersonDetector()
    trk  = IOUTracker(iou_thresh=0.4, max_age=30)
    adeng= AdEngine()

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[retail-edge] Failed to open source: {args.source}")
        return

    ok, f0 = cap.read()
    if not ok or f0 is None:
        print("[retail-edge] No frames available.")
        return

    heat = HeatmapTracker(f0.shape, cam_id=cam_id, dwell_thresh=DWELL_THRESH, hot_thresh=HOT_THRESH, out_dir=HEAT_OUT_DIR)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except:
        pass

    # decision window accumulators
    totals = {"faces": 0, "age_sum": 0.0, "male": 0, "female": 0}
    age_ema = None  # smoothed age preview
    window_start_ts = time.time()

    frame_index = 0
    cam_win = "camera"
    ad = AdPlayer(win_name="video")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # frame skip
        if FSKIP and (frame_index % (FSKIP + 1) != 0):
            frame_index += 1
            # show frames minimally so UI stays responsive
            if not args.no_display:
                cv2.imshow(cam_win, frame)
                if ad.active():
                    ad.step()  # keep ad flowing
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')): break
                if key in (ord('c'),): ad.stop()
            continue

        # ----- Person detection / tracking / heatmap -----
        dets = det.infer(frame) or []
        dets = [d for d in dets if d.get('conf', 1.0) >= MIN_SCORE]
        tracks = trk.update(dets)
        heat.update(frame, tracks)

        # ----- Face detect + age/gender (EV + confidence) -----
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_det(gray, 0)

        faces_for_fusion = []
        for r in rects:
            x, y, w, h = r.left(), r.top(), r.width(), r.height()
            x, y, w, h = clip_roi(x, y, w, h, W, H)
            x1, y1, x2, y2 = x, y, x + w, y + h
            sub = frame[y1:y2, x1:x2]
            age_ev, gender_lbl, gender_conf = None, None, None

            if sub.size > 0 and (gender_net is not None and age_net is not None):
                blob = cv2.dnn.blobFromImage(
                    sub, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                # gender
                gender_net.setInput(blob)
                g = gender_net.forward()[0]  # logits or probs [Male, Female]
                gp = softmax(g)
                if gp[0] >= gp[1]:
                    gender_lbl = "Male"
                    gender_conf = float(gp[0])
                else:
                    gender_lbl = "Female"
                    gender_conf = float(gp[1])
                # age
                age_net.setInput(blob)
                a = age_net.forward()[0]  # 8-way logits or probs
                age_ev = age_expected_value(a, AGE_MIDS)

            # apply gender confidence threshold
            if gender_conf is not None and gender_conf < GENDER_CONF_THR:
                gender_lbl = "unsure"

            # EMA smoothing for age (global preview)
            if age_ev is not None:
                age_ema = age_ev if age_ema is None else (EMA_ALPHA * age_ev + (1 - EMA_ALPHA) * age_ema)

            # Add to fusion list
            faces_for_fusion.append({
                "box": (x1, y1, x2, y2),
                "age": int(round(age_ev)) if age_ev is not None else None,
                "gender": gender_lbl,
                "emotion": None
            })

        # Fuse faces to tracks (ID-aware overlay)
        enriched = fuse(tracks, faces_for_fusion)

        # update window totals for decisioning
        for f in faces_for_fusion:
            g = (f.get("gender") or "").lower()
            if g == "male": totals["male"] += 1
            elif g == "female": totals["female"] += 1
            if f.get("age") is not None: totals["age_sum"] += float(f["age"])
            totals["faces"] += 1

        # ----- Render heatmap overlay -----
        overlay = heat.render(frame)

        # ----- Decision window (only when NO ad is running) -----
        now = time.time()
        window_elapsed = (now - window_start_ts) >= WINDOW_SEC
        if (not ad.active()) and window_elapsed:
            if totals["faces"] > 0:
                avg_age = totals["age_sum"] / totals["faces"]
                frac_male = totals["male"] / max(1, totals["faces"])
                ad_path = adeng.choose(avg_age, frac_male)  # personas.yaml mapping
                if ad_path:
                    ad.start(ad_path)
                print(f"[retail-edge] Decision @ {time.strftime('%H:%M:%S')}: "
                      f"avg_age={avg_age:.1f} (EMA={age_ema:.1f} if set) male%={frac_male*100:.1f} -> {ad_path}")
            # reset window
            totals = {"faces": 0, "age_sum": 0.0, "male": 0, "female": 0}
            window_start_ts = now

        # ----- Display -----
        if not args.no_display:
            vis = overlay.copy()

            # Draw enriched tracks
            for t in enriched:
                x1, y1, x2, y2 = map(int, t['box'])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {t['id']}"
                g = t.get('gender')
                a = t.get('age')
                if g or a:
                    label += f" | {(g if g else '?')} {(a if a is not None else '')}"
                cv2.putText(vis, label, (x1, max(8, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # HUD with EMA age
            if age_ema is not None:
                txt = f"Age(EMA): {age_ema:.1f}"
                cv2.putText(vis, txt, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow(cam_win, vis)

            # Step ad window (non-blocking)
            if ad.active():
                ad.step()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # Esc or q
                break
            if key in (ord('c'),):     # cancel ad
                ad.stop()

        frame_index += 1

    # ----- Save heatmap outputs -----
    try:
        report_path = heat.save_report()
        hm = heat.heatmap.copy()
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        out_png = os.path.join(HEAT_OUT_DIR, f"{cam_id}_heatmap.png")
        cv2.imwrite(out_png, hm_color)
        print(f"[retail-edge] Saved report: {report_path}")
        print(f"[retail-edge] Saved heatmap: {out_png}")
    except Exception as e:
        print(f"[retail-edge] Failed to save report: {e}")

    # Cleanup
    ad.stop()
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
