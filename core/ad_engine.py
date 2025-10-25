import yaml, os

class AdEngine:
    def __init__(self, personas_yaml='config/personas.yaml'):
        with open(personas_yaml, 'r') as f:
            data = yaml.safe_load(f) or {}
        self.rules = data.get('personas', [])
        self.default_ad = (data.get('defaults') or {}).get('ad', None)

    def choose(self, avg_age, frac_male):
        gender = 'male' if frac_male >= 0.5 else 'female'
        for r in self.rules:
            w = r.get('when', {})
            min_age = w.get('min_age', -1e9)
            max_age = w.get('max_age',  1e9)
            g      = (w.get('gender') or '').lower() or None
            if (avg_age >= min_age) and (avg_age <= max_age) and (g in (None, gender)):
                return r.get('ad')
        return self.default_ad
