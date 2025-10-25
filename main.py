import os, sys, time, platform, argparse, numpy as np, cv2, dlib
from core.ad_player import AdPlayer
from core.heatmap import Heatmap
from core.utils import (
    must_exist, age_expected_value, clip_roi, rect_to_xywh, ts_now,
)

# ----------------- CONFIG: paths (env overrides allowed) -----------------
GENDER_PROTOTXT = os.getenv('GENDER_PROTOTXT', 'mods/gender_deploy.prototxt')
GENDER_CAFFE    = os.getenv('GENDER_MODEL',    'mods/gender_net.caffemodel')
AGE_PROTOTXT    = os.getenv('AGE_PROTOTXT',    'mods/age_deploy.prototxt')
AGE_CAFFE       = os.getenv('AGE_MODEL',       'mods/age_net.caffemodel')

# Example videos (make sure they exist)
VIDEO_KIDS_M   = os.getenv('VIDEO_KIDS_M',  'videos/kids_male_video.mp4')
VIDEO_KIDS_F   = os.getenv('VIDEO_KIDS_F',  'videos/kids_female_video.mp4')
VIDEO_ADULT_M  = os.getenv('VIDEO_ADULT_M', 'videos/adult_male_video.mp4')
VIDEO_ADULT_F  = os.getenv('VIDEO_ADULT_F', 'videos/adult_female_video.mp4')

# Thresholds & constants
GENDER_CONF_THR = float(os.getenv('GENDER_CONF_THR', '0.90'))
EMA_ALPHA       = float(os.getenv('EMA_ALPHA', '0.6'))
WINDOW_SEC      = float(os.getenv('WINDOW_SEC', '3.0'))

def parse_args():
    ap = argparse.ArgumentParser("Retail demo: face attrs + ads + heatmap (modular)")
    ap.add_argument("--source", type=str, default="0", help="webcam index or video path")
    ap.add_argument("--heatmap-dir", type=str, default=os.getenv("HEATMAP_DIR", "heatmap_reports"),
                    help="output folder for heatmap PNG/JSON")
    ap.add_argument("--cam-id", type=str, default="cam0", help="camera id used in filenames")
    return ap.parse_args()

def main():
    args = parse_args()

    # Check model files
    ok = True
    ok &= must_exist(GENDER_PROTOTXT, 'GENDER_PROTOTXT')
    ok &= must_exist(GENDER_CAFFE,    'GENDER_CAFFE')
    ok &= must_exist(AGE_PROTOTXT,    'AGE_PROTOTXT')
    ok &= must_exist(AGE_CAFFE,       'AGE_CAFFE')
    if not ok:
        print("[X] Fix missing model files above and rerun.")
        sys.exit(1)

    # Load models
    try:
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT, GENDER_CAFFE)
        age_net    = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT,    AGE_CAFFE)
    except Exception as e:
        print("[X] Failed to load Caffe models:", e)
        sys.exit(1)

    # Face detector
    try:
        detector = dlib.get_frontal_face_detector()
    except Exception as e:
        print("[X] Dlib face detector failed to initialize:", e)
        sys.exit(1)

    # Open source
    src = args.source
    if src.isdigit():
        cap_flag = cv2.CAP_DSHOW if platform.system() == 'Windows' else 0
        cap = cv2.VideoCapture(int(src), cap_flag) if platform.system() == 'Windows' else cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[X] Could not open source: {src}")
        sys.exit(1)

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)

    # Modules
    ad   = AdPlayer(win='video')
    heat = Heatmap(alpha=float(os.getenv('HEAT_ALPHA', '0.45')))

    # Stats for decisioning
    ema_age = None
    total_faces = 0
    total_age = 0.0
    total_male = 0
    total_female = 0
    window_start = time.time()

    print("[✓] Running. Keys: 'q' quit | 'c' cancel ad | 'h' toggle heatmap | 'r' reset heatmap | 's' save now")

    while True:
        ok, frame = cap.read()
        if not ok:
            if src.isdigit():  # camera: retry
                time.sleep(0.02)
                continue
            print("[i] End of file / no more frames.")
            break

        H, W = frame.shape[:2]
        heat.ensure((H, W))
        heat.decay()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for r in rects:
            x, y, w, h = rect_to_xywh(r)
            if w <= 0 or h <= 0:
                continue

            # Heat contribution
            heat.add_face(frame.shape, x, y, w, h)

            # Model input (padded crop)
            face_roi, (fx, fy, fw, fh) = clip_roi(frame, x, y, w, h, pad=0.20)
            if face_roi is None or face_roi.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(
                face_roi, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False, crop=False
            )

            # Gender
            gender_net.setInput(blob)
            g = gender_net.forward()[0]
            g_idx = int(np.argmax(g))
            g_conf = float(g[g_idx])
            gender = 'Male' if g_idx == 0 else 'Female'

            # Age (expected value with EMA)
            age_net.setInput(blob)
            a = age_net.forward()[0]
            age_ev = age_expected_value(a)
            ema_age = age_ev if ema_age is None else (EMA_ALPHA*age_ev + (1-EMA_ALPHA)*ema_age)

            # Overlay
            cv2.rectangle(frame, (fy and fx), (fx+fw, fy+fh), (0,255,0), 2)  # (x,y) → (fx,fy)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)
            g_text = gender if g_conf >= GENDER_CONF_THR else "Gender: unsure"
            a_text = f"Age≈ {int(round(ema_age))}"
            cv2.putText(frame, g_text, (fx, max(0, fy-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, a_text, (fx, fy+fh+22),        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Stats
            total_faces += 1
            total_age += age_ev
            if g_conf >= GENDER_CONF_THR:
                if gender == 'Male': total_male += 1
                else:                total_female += 1

        # Render heatmap overlay
        frame_show = heat.overlay(frame)
        cv2.imshow('camera', frame_show)

        # Non-blocking ad
        if ad.active():
            ad_frame = ad.step()
            if ad_frame is not None:
                cv2.imshow('video', ad_frame)
        else:
            if (time.time() - window_start) > WINDOW_SEC:
                if total_faces > 0:
                    avg_age = total_age / total_faces
                    votes = total_male + total_female
                    male_ratio = (total_male / votes) if votes > 0 else 0.5

                    if avg_age <= 12:
                        path = VIDEO_KIDS_M if male_ratio >= 0.5 else VIDEO_KIDS_F
                    else:
                        path = VIDEO_ADULT_M if male_ratio >= 0.5 else VIDEO_ADULT_F

                    print(f"[i] {ts_now()} starting ad: {path}")
                    ad.start(path)

                # reset stats
                total_faces = total_male = total_female = 0
                total_age = 0.0
                window_start = time.time()

        # Keys
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            ad.stop()
        elif k == ord('h'):
            heat.enabled = not heat.enabled
            print(f"[i] Heatmap {'ON' if heat.enabled else 'OFF'}")
        elif k == ord('r'):
            heat.reset()
            print("[i] Heatmap reset.")
        elif k == ord('s'):
            heat.save(args.heatmap_dir, cam_id=args.cam_id)

    # Cleanup & auto-save
    ad.stop()
    cap.release()
    if heat.acc is not None:
        print("[i] Saving final heatmap…")
        heat.save(args.heatmap_dir, cam_id=args.cam_id)
    cv2.destroyAllWindows()
    print("[✓] Clean exit.")

if __name__ == "__main__":
    main()
