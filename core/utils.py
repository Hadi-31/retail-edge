import os, datetime, numpy as np

AGE_MIDS  = np.array([1, 5, 10, 18, 28, 40.5, 50.5, 65], dtype=np.float32)

def must_exist(path, label):
    if not os.path.isfile(path):
        print(f"[!] Missing {label}: {path}")
        return False
    return True

def age_expected_value(age_logits):
    p = age_logits.flatten()
    p = np.exp(p - p.max()); p = p / max(p.sum(), 1e-8)
    return float((p * AGE_MIDS).sum())

def rect_to_xywh(r):
    x, y = r.left(), r.top()
    w, h = r.right() - r.left(), r.bottom() - r.top()
    return x, y, w, h

def clip_roi(img, x, y, w, h, pad=0.2):
    H, W = img.shape[:2]
    px, py = int(w*pad), int(h*pad)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(W, x + w + px)
    y1 = min(H, y + h + py)
    if x1 <= x0 or y1 <= y0:
        return None, (x, y, w, h)
    return img[y0:y1, x0:x1], (x0, y0, x1-x0, y1-y0)

def ts_now():
    return datetime.datetime.now().strftime('%H:%M:%S')
