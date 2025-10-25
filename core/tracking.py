from .utils import iou

class IOUTracker:
    def __init__(self, iou_thresh=0.4, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}    # id -> {'box':(x1,y1,x2,y2), 'age':0, 'hits':0}
        self._next_id = 1

    def update(self, detections):
        # detections: [{'box':(x1,y1,x2,y2),'conf':0.9}, ...]
        assigned = set()
        # age tracks
        for tid, t in list(self.tracks.items()):
            t['age'] += 1
            if t['age'] > self.max_age:
                del self.tracks[tid]

        # greedy match by IOU
        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                ov = iou(det['box'], t['box'])
                if ov > best_iou:
                    best_iou, best_id = ov, tid
            if best_iou >= self.iou_thresh and best_id is not None:
                self.tracks[best_id]['box'] = det['box']
                self.tracks[best_id]['age'] = 0
                self.tracks[best_id]['hits'] = self.tracks[best_id].get('hits',0)+1
                assigned.add(best_id)
            else:
                tid = self._next_id; self._next_id += 1
                self.tracks[tid] = {'box': det['box'], 'age': 0, 'hits': 1}

        # output list
        out = []
        for tid, t in self.tracks.items():
            out.append({'id': tid, 'box': t['box']})
        return out
