"""Small bay checks for the round-2 audit: distances between the station target coordinates and the selected model
cells (deepest wet cell within +-0.022 deg), and the sensitivity of the simultaneous tidal fit to its window
(days 3-33 vs 8-33). Merged into checks.json."""
import os, paths as P
import json
import numpy as np
D = P.SFBAY
st = np.load(f"{D}/sfbay_stations.npz", allow_pickle=True); td = np.load(f"{D}/sfbay_tides.npz")
TARGET = {"Oakland": (37.81, -122.34), "Richmond": (37.93, -122.42), "San Pablo Bay": (38.05, -122.42)}
KM = 111.0 * np.cos(np.radians(37.9))
out = {"station_distance_km": {}}
for nm, la, lo in zip(st["names"], st["lat"], st["lon"]):
    nm = str(nm)
    if nm in TARGET:
        tl, tg = TARGET[nm]
        out["station_distance_km"][nm] = float(np.hypot((lo - tg) * KM, (la - tl) * 111.0))
DEG = np.pi / 180
SPEEDS = {"M2": 28.98410, "S2": 30.0, "N2": 28.43973, "K1": 15.04107, "O1": 13.94304, "P1": 14.95893, "M4": 57.96821, "MS4": 58.98410}
def fit(t, y):
    cols = [np.ones_like(t)]
    for s in SPEEDS.values():
        w = s * DEG / 3600; cols += [np.cos(w * t), np.sin(w * t)]
    c, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    return {n: float(np.hypot(c[1 + 2 * i], c[2 + 2 * i])) for i, n in enumerate(SPEEDS) if i < 6}
t, e = td["t"], td["eta_gg"]
a = fit(t[t >= 3 * 86400], e[t >= 3 * 86400]); b = fit(t[t >= 8 * 86400], e[t >= 8 * 86400])
out["fit_window_amp_days3_33"] = a; out["fit_window_amp_days8_33"] = b
out["fit_window_max_rel_change"] = max(abs(b[n] / a[n] - 1) for n in a)
ck = json.load(open("checks.json")); ck.update(out); json.dump(ck, open("checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
