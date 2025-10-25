import os, json, datetime, cv2, numpy as np

HEAT_DECAY     = float(os.getenv('HEAT_DECAY', '0.97'))
HEAT_STRENGTH  = float(os.getenv('HEAT_STRENGTH', '1.0'))
HEAT_BLUR      = int(os.getenv('HEAT_BLUR', '31'))
HEAT_ALPHA     = float(os.getenv('HEAT_ALPHA', '0.45'))
ELLIPSE_SCALE  = float(os.getenv('ELLIPSE_SCALE', '0.6'))
GRID_DOWNSAMPLE= os.getenv('GRID_DOWNSAMPLE', '')  # e.g. "64"

def _ts():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

class Heatmap:
    """
    Pixel-accurate accumulator with decay. Saves PNG + JSON to a folder.
    Optionally includes a compact downsampled grid in JSON via GRID_DOWNSAMPLE.
    """
    def __init__(self, alpha=HEAT_ALPHA):
        self.acc = None   # float32 HxW
        self.enabled = True
        self.alpha = alpha

    def ensure(self, shape_hw):
        h, w = shape_hw
        if self.acc is None or self.acc.shape != (h, w):
            self.acc = np.zeros((h, w), dtype=np.float32)

    def decay(self):
        if self.acc is not None:
            self.acc *= HEAT_DECAY

    def add_face(self, frame_shape, x, y, w, h):
        H, W = frame_shape[:2]
        self.ensure((H, W))
        cx = int(x + w * 0.5)
        cy = int(y + h * 0.5)
        ax = max(3, int(w * 0.5 * ELLIPSE_SCALE))
        ay = max(3, int(h * 0.5 * ELLIPSE_SCALE))
        mask = np.zeros((H, W), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
        if HEAT_BLUR and HEAT_BLUR > 1 and HEAT_BLUR % 2 == 1:
            mask = cv2.GaussianBlur(mask, (HEAT_BLUR, HEAT_BLUR), 0)
        self.acc += (mask * HEAT_STRENGTH)

    def overlay(self, frame_bgr):
        if not self.enabled or self.acc is None:
            return frame_bgr
        hm = self.acc.copy()
        if hm.max() > 0:
            hm = (hm / hm.max()) * 255.0
        hm = hm.astype(np.uint8)
        color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame_bgr, 1.0, color, self.alpha, 0)

    def _downsample_grid(self, grid):
        if not grid: return None
        grid = int(grid)
        H, W = self.acc.shape
        return cv2.resize(self.acc, (grid, grid), interpolation=cv2.INTER_AREA)

    def save(self, out_dir, cam_id="cam0", normalize_png=True):
        if self.acc is None:
            print("[i] Heatmap empty; nothing to save.")
            return None, None

        os.makedirs(out_dir, exist_ok=True)
        base = f"{cam_id}_{_ts()}"
        json_path = os.path.join(out_dir, f"{base}_heatmap.json")
        png_path  = os.path.join(out_dir, f"{base}_heatmap.png")

        payload = {
            "camera": str(cam_id),
            "generated_at": datetime.datetime.now().isoformat(timespec='seconds'),
            "height": int(self.acc.shape[0]),
            "width": int(self.acc.shape[1]),
            "heatmap_seconds": self.acc.tolist(),
        }
        if GRID_DOWNSAMPLE and GRID_DOWNSAMPLE.isdigit():
            grid = self._downsample_grid(GRID_DOWNSAMPLE)
            payload["grid"] = int(GRID_DOWNSAMPLE)
            payload["heatmap_seconds_grid"] = grid.tolist()

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        hm = self.acc
        if normalize_png:
            hm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
        hm_u8 = hm.astype('uint8')
        color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        cv2.imwrite(png_path, color)

        print(f"[✓] Heatmap JSON: {json_path}")
        print(f"[✓] Heatmap PNG : {png_path}")
        return json_path, png_path

    def reset(self):
        if self.acc is not None:
            self.acc.fill(0)
