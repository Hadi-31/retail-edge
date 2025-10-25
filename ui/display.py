import cv2
from ..core.utils import draw_box

def render(frame, tracks, enriched, show_boxes=True, ad_thumb=None, heatmap_overlay=None):
    out = frame.copy()
    if heatmap_overlay is not None:
        out = heatmap_overlay
    if show_boxes:
        for t in tracks:
            label = f"ID {t['id']}"
            for e in enriched:
                if e['id'] == t['id']:
                    g = e.get('gender')
                    a = e.get('age')
                    if g or a:
                        label += f" | {g or '?'} {a or ''}"
                    break
            draw_box(out, t['box'], label=label)
    if ad_thumb is not None:
        h, w = out.shape[:2]
        th = min(160, h//4)
        tw = int(th * (ad_thumb.shape[1]/ad_thumb.shape[0]))
        thumb = cv2.resize(ad_thumb, (tw, th))
        out[10:10+th, 10:10+tw] = thumb
    return out
