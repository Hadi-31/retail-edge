# 🛍️ retail-edge

Lightweight, edge-ready **retail analytics and ad targeting system**.  
Detects faces, estimates **age** and **gender**, plays **targeted ads**, and builds a **real-time heatmap** of customer attention.  
All modules are self-contained and built with **OpenCV** and **dlib**, optimized for real-time edge devices like Jetson or Raspberry Pi.

---

## ✨ Features
- Real-time **face detection** using `dlib`
- **Age & gender estimation** via OpenCV DNN (Caffe models)
- Automatic **ad selection** (kids/adults × male/female)
- Persistent **heatmap visualization** of detected faces
- Saves **PNG + JSON heatmaps** under `heatmap_reports/`
- Modular, easy to extend or integrate with other systems

---

## 🧩 Project Structure
```
retail-edge/
├─ main.py
├─ core/
│  ├─ heatmap.py
│  ├─ ad_player.py
│  └─ utils.py
├─ mods/
│  ├─ gender_deploy.prototxt
│  ├─ gender_net.caffemodel
│  ├─ age_deploy.prototxt
│  ├─ age_net.caffemodel
│  └─ shape_predictor_68_face_landmarks.dat
├─ videos/
│  ├─ kids_male_video.mp4
│  ├─ kids_female_video.mp4
│  ├─ adult_male_video.mp4
│  └─ adult_female_video.mp4
├─ heatmap_reports/
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Installation
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt**
```txt
numpy==1.26.4
opencv-python
dlib
pillow
PyYAML
tqdm
```

> 💡 Note: `dlib` may require build tools (Visual Studio on Windows, Xcode CLI on macOS, or `cmake` + `build-essential` on Linux).

---

## ▶️ Usage
```bash
python main.py                # Use webcam
python main.py --source path/to/video.mp4
```

---

## 🎛️ Keyboard Controls
| Key | Action |
|-----|---------|
| q | Quit |
| s | Save heatmap |
| r | Reset heatmap |
| h | Toggle heatmap overlay |
| c | Cancel ad playback |

---

## 🌡️ Heatmap Output
Each session generates:
```
heatmap_reports/
├─ cam0_YYYYMMDD_HHMMSS_heatmap.png
└─ cam0_YYYYMMDD_HHMMSS_heatmap.json
```

---

## 🧠 Model Files
Place your model files in `mods/`:
```
mods/
├─ gender_deploy.prototxt
├─ gender_net.caffemodel
├─ age_deploy.prototxt
├─ age_net.caffemodel
└─ shape_predictor_68_face_landmarks.dat
```

---

## 💾 Notes
- Fully offline after models/videos are downloaded.
- Optimized for edge devices.
- Easy to extend with cloud sync or MQTT.

---

## 📄 License
MIT License © 2025
