\
import os, cv2, argparse, time, numpy as np
from apps.person_detect import PersonDetector
from apps.face_attrs import FaceAttrEstimator
from core.tracking import IOUTracker
from core.heatmap_tracker import HeatmapTracker
from core.fusion import fuse
from core.ad_engine import AdEngine

# Env
FSKIP         = int(os.getenv("FRAME_SKIP", "0"))
MIN_SCORE     = float(os.getenv("MIN_SCORE", "0.30"))
DWELL_THRESH  = float(os.getenv("DWELL_THRESH", "5"))
HOT_THRESH    = float(os.getenv("HOT_THRESH", "10"))
HEAT_OUT_DIR  = os.getenv("HEAT_OUT_DIR", "heatmap_reports")
DRAW_BOXES    = os.getenv("DRAW_BOXES", "1") == "1"

def parse_args():
    ap = argparse.ArgumentParser("retail-edge: tracking + demographics + ads + heatmap")
    ap.add_argument("--source", type=str, default="0", help="webcam index or video file")
    ap.add_argument("--no-display", action="store_true", help="run headless")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between ad decisions")
    ap.add_argument("--kids-age", type=int, default=12, help="children threshold for rules")
    ap.add_argument("--videos-dir", type=str, default="ui/assets/ads", help="ad assets directory")
    return ap.parse_args()

def load_ad(asset_path):
    if not asset_path or not os.path.exists(asset_path):
        return None
    cap = cv2.VideoCapture(asset_path)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        return frame
    # try image
    img = cv2.imread(asset_path)
    return img

def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    cam_id = str(args.source)

    det = PersonDetector()
    face = FaceAttrEstimator()
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
    try: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except: pass

    totals = {"faces":0, "age":0.0, "male":0, "female":0}
    last_decision_ts = time.time()
    frame_index = 0
    ad_frame = None
    win = f"retail-edge [{cam_id}]"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if FSKIP and (frame_index % (FSKIP + 1) != 0):
            frame_index += 1
            if not args.no_display:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
            continue

        dets = det.infer(frame) or []
        dets = [d for d in dets if d.get('conf', 1.0) >= MIN_SCORE]
        tracks = trk.update(dets)
        heat.update(frame, tracks)

        faces = face.infer(frame) or []
        for f in faces:
            if f.get('gender') == 'Male': totals["male"] += 1
            if f.get('gender') == 'Female': totals["female"] += 1
            if f.get('age') is not None: totals["age"] += float(f['age'])
            totals["faces"] += 1

        enriched = fuse(tracks, faces)

        overlay = heat.render(frame)

        # ad decision
        now = time.time()
        if (now - last_decision_ts) >= args.interval:
            if totals["faces"] > 0:
                avg_age = totals["age"] / totals["faces"]
                frac_male = totals["male"] / totals["faces"]
                ad_path = adeng.choose(avg_age, frac_male)
                ad_frame = load_ad(ad_path) if ad_path else None
                print(f"[retail-edge] Decision: avg_age={avg_age:.1f} male%={frac_male*100:.1f} -> {ad_path}")
            totals = {"faces":0, "age":0.0, "male":0, "female":0}
            last_decision_ts = now

        if not args.no_display:
            vis = overlay.copy()
            if ad_frame is not None:
                h, w = vis.shape[:2]
                th = min(200, h//3)
                tw = int(th * (ad_frame.shape[1]/ad_frame.shape[0]))
                thumb = cv2.resize(ad_frame, (tw, th))
                vis[10:10+th, 10:10+tw] = thumb
            for t in enriched:
                label = f"ID {t['id']}"
                if t.get('gender') or t.get('age'):
                    label += f" | {t.get('gender','?')} {t.get('age') or ''}"
                x1,y1,x2,y2 = map(int, t['box'])
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, label, (x1, max(8,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            cv2.imshow(win, vis)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break

        frame_index += 1

    try:
        report_path = heat.save_report()
        # also dump heatmap png
        hm = heat.heatmap.copy()
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        out_png = os.path.join(HEAT_OUT_DIR, f"{cam_id}_heatmap.png")
        cv2.imwrite(out_png, hm_color)
        print(f"[retail-edge] Saved report: {report_path}")
        print(f"[retail-edge] Saved heatmap: {out_png}")
    except Exception as e:
        print(f"[retail-edge] Failed to save report: {e}")

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
