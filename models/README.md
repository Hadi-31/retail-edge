# Models

Place your model files here.

## Age/Gender (OpenCV DNN Caffe)
- `gender_deploy.prototxt`
- `gender_net.caffemodel`
- `age_deploy.prototxt`
- `age_net.caffemodel`

Set environment variables to point to these paths if you rename or move them.

## Person Detection
- You can use the built-in HOG fallback (no extra files).
- For higher accuracy/speed, bring your own ONNX/DFP YOLO and integrate in `apps/person_detect.py`.
