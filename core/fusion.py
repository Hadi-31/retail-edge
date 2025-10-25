from .utils import iou

def fuse(tracks, faces, iou_thresh=0.3):
    # tracks: [{'id', 'box'}]; faces: [{'box','age','gender','emotion'}]
    enriched = []
    for t in tracks:
        best = None
        best_i = 0.0
        for f in faces:
            ov = iou(t['box'], f['box'])
            if ov > best_i:
                best_i, best = ov, f
        rec = dict(t)
        if best and best_i >= iou_thresh:
            rec.update({k: best.get(k) for k in ('age','gender','emotion')})
        enriched.append(rec)
    return enriched
