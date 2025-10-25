# retail-edge

Edge-ready retail analytics and ad targeting demo. Combines **person detection + tracking**, **face attributes (age/gender/emotion)**, and a **dwell-time heatmap**. Periodically chooses an ad (image/video) based on detected demographics and logs heatmap reports per camera. 

## Features
- Person detection (YOLO or HOG fallback).
- IOU-based multi-object tracking with persistent IDs.
- (Optional) Face attributes via DeepFace or OpenCV DNN (age/gender).
- Dwell-time **heatmap** per camera, plus an aggregator across cameras.
- Rule-based **ad engine** driven by `config/personas.yaml`.
- Overlay UI: boxes, labels, ad thumbnail, optional heatmap blend.
- Saves `*.json` reports and a `*.png` heatmap at the end of each run.

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> Note: `dlib` may require build tools. If installation is difficult on your platform, you can comment out dlib usage in `apps/face_attrs.py` and rely on DeepFace-only (or vice versa).

## Run
```bash
python main.py --source 0                      # webcam
python main.py --source data/clip.mp4          # a file
```

### Helpful flags
- `--no-display` – headless mode (still writes reports & heatmap PNG)
- `--interval 3.0` – seconds between demographic-based ad decisions
- `--kids-age 12` – threshold between kids vs adults
- `--videos-dir ui/assets/ads` – where ad assets live

### Environment variables
```
FRAME_SKIP=0        # skip frames to save compute (0 → process all)
MIN_SCORE=0.30      # detector confidence threshold
DWELL_THRESH=5      # seconds to count a dwell visit (heatmap)
HOT_THRESH=10       # seconds to mark as hotspot (heatmap)
HEAT_OUT_DIR=heatmap_reports
DRAW_BOXES=1        # draw tracked boxes

# DNN model files (if using OpenCV-based age/gender)
GENDER_PROTOTXT=models/gender_deploy.prototxt
GENDER_MODEL=models/gender_net.caffemodel
AGE_PROTOTXT=models/age_deploy.prototxt
AGE_MODEL=models/age_net.caffemodel
GENDER_CONF_THRESH=0.5
AGE_CONF_THRESH=0.9
```

## Models
Place any model files under `models/`. See `models/README.md` for notes.

## Output
Per run, you’ll find in `heatmap_reports/`:
- `{cam}_heatmap.json` – dwell grid with timestamps and settings
- `{cam}_heatmap.png` – colorized heatmap image
- `master_heatmap.json` – global merge across runs/cameras (via `core/heatmap_aggregator.py`)

## Project structure
```
retail-edge/
├─ main.py
├─ requirements.txt
├─ README.md
├─ apps/
│  ├─ person_detect.py
│  └─ face_attrs.py
├─ core/
│  ├─ tracking.py
│  ├─ fusion.py
│  ├─ ad_engine.py
│  ├─ utils.py
│  ├─ heatmap_tracker.py
│  └─ heatmap_aggregator.py
├─ ui/
│  ├─ display.py
│  └─ assets/ads/   # add your images/videos here
├─ config/
│  └─ personas.yaml
├─ models/
│  └─ README.md
└─ heatmap_reports/
```

## Notes
- If you don’t have DeepFace models cached, the first run may download them.
- If you don’t have Caffe age/gender models, the pipeline still runs (attributes become optional).
- For best performance, supply a proper person detector model (YOLO/ONNX/DFP) and GPU-accelerated OpenCV.
