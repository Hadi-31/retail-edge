
import os, json, cv2, numpy as np, time

class SimpleHeatmap:
    """Ultra-simple time-weighted heatmap.
    - grid: heat accumulation grid size (grid x grid)
    - update(tracks, dt): add dt seconds to the cell under each track's center
    - render(frame, alpha): overlay normalized heat on the frame
    - save(out_dir, cam_id): write JSON and PNG
    """
    def __init__(self, frame_shape, grid=64):
        self.h, self.w = frame_shape[:2]
        self.grid = int(grid)
        self.cell_h = max(1, self.h // self.grid)
        self.cell_w = max(1, self.w // self.grid)
        self.buf = np.zeros((self.grid, self.grid), dtype=np.float32)
        self._last_ts = time.time()

    def update(self, tracks):
        now = time.time()
        dt = max(0.0, now - self._last_ts)
        self._last_ts = now
        if dt <= 0: 
            return
        for t in tracks or []:
            x1,y1,x2,y2 = map(int, t['box'])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cx = min(max(cx, 0), self.w - 1)
            cy = min(max(cy, 0), self.h - 1)
            gi = min(self.grid - 1, cx // self.cell_w)
            gj = min(self.grid - 1, cy // self.cell_h)
            self.buf[gj, gi] += dt

    def render(self, frame, alpha=0.45):
        hm = self.buf
        if hm.max() > 0:
            hm_norm = (hm / hm.max() * 255).astype('uint8')
        else:
            hm_norm = hm.astype('uint8')
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_up = cv2.resize(hm_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(frame, 1.0, hm_up, alpha, 0)
        return out

    def save(self, out_dir, cam_id):
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f"{cam_id}_heatmap.json")
        with open(json_path, "w") as f:
            json.dump({
                "camera": str(cam_id),
                "grid": int(self.grid),
                "heatmap_seconds": self.buf.tolist(),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        # PNG
        hm = self.buf
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        png_path = os.path.join(out_dir, f"{cam_id}_heatmap.png")
        cv2.imwrite(png_path, hm_color)
        return json_path, png_path
