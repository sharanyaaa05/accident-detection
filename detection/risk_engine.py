WEIGHTS = {
    "jaywalking": 0.25,
    "speeding": 0.20,
    "phone": 0.15,
    "drowsy": 0.30,
    "signal violation": 0.10,
    "accident": 1.00
}

def compute_risk(events):
    return round(sum(WEIGHTS[e] for e in events if e in WEIGHTS), 2)
