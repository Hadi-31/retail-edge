import os, json, cv2, numpy as np, time

class HeatmapTracker:
    def __init__(self, frame_shape, cam_id='cam0', dwell_thresh=5, hot_thresh=10, out_dir='heatmap_reports', grid=64):
        self.h, self.w = frame_shape[:2]
        self.grid = grid
        self.cell_h = max(1, self.h // grid)
        self.cell_w = max(1, self.w // grid)
        self.heatmap = np.zeros((grid, grid), dtype=np.float32)
        self.cam_id = cam_id
        self.dwell_thresh = dwell_thresh
        self.hot_thresh = hot_thresh
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._last_ts = time.time()

    def update(self, frame, tracked):
        # accumulate time-based occupancy per cell using center of track boxes
        now = time.time()
        dt = max(0.0, now - self._last_ts)
        self._last_ts = now
        for t in tracked:
            x1,y1,x2,y2 = map(int, t['box'])
            cx = max(0, min(self.w-1, (x1+x2)//2))
            cy = max(0, min(self.h-1, (y1+y2)//2))
            j = min(self.grid-1, cy // self.cell_h)
            i = min(self.grid-1, cx // self.cell_w)
            self.heatmap[j,i] += dt

    def render(self, frame, alpha=0.45):
        # normalize and overlay
        hm = self.heatmap.copy()
        if hm.max() > 0:
            hm_norm = (hm / hm.max() * 255).astype('uint8')
        else:
            hm_norm = hm.astype('uint8')
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_up = cv2.resize(hm_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(frame, 1.0, hm_up, alpha, 0)
        return out

    def save_report(self):
        report = {
            'camera': self.cam_id,
            'grid': int(self.grid),
            'dwell_thresh': float(self.dwell_thresh),
            'hot_thresh': float(self.hot_thresh),
            'heatmap_seconds': self.heatmap.tolist(),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = os.path.join(self.out_dir, f'{self.cam_id}_heatmap.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        return path
