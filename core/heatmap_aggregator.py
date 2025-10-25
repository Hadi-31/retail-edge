import os, json, numpy as np

def aggregate_reports(report_dir='heatmap_reports', out_name='master_heatmap.json'):
    grids = []
    meta = []
    for fn in os.listdir(report_dir):
        if fn.endswith('_heatmap.json'):
            with open(os.path.join(report_dir, fn), 'r') as f:
                data = json.load(f)
                grids.append(np.array(data['heatmap_seconds'], dtype=float))
                meta.append({'camera': data.get('camera'), 'grid': data.get('grid')})
    if not grids:
        return None
    base = grids[0]
    for g in grids[1:]:
        base = base + g
    out = {
        'grid': int(meta[0]['grid'] if meta else len(base)),
        'heatmap_seconds': base.tolist(),
        'sources': meta,
    }
    out_path = os.path.join(report_dir, out_name)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    return out_path
