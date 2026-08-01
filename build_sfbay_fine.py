"""Regrid the NOAA Coastal Relief Model (CRM Vol.7, 3 arc-sec) subset of San
Francisco Bay onto a uniform ~200 m grid for the semi-implicit tidal solver.
Keeps only water connected to the Pacific open boundary (flood fill)."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage

RAW = "sfbay_fine/crm_raw.npz"
OUT = "sfbay_fine/sfbay_fine.npz"
TARGET_M = 200.0  # target cell size (m)

d = np.load(RAW); zlat, zlon, z = d["lat"], d["lon"], d["z"]
lat0, lat1 = float(zlat.min()), float(zlat.max())
lon0, lon1 = float(zlon.min()), float(zlon.max())
latm = 0.5*(lat0+lat1)
m_per_deg_lat = 111320.0
m_per_deg_lon = 111320.0*np.cos(np.radians(latm))
dlat = TARGET_M/m_per_deg_lat
dlon = TARGET_M/m_per_deg_lon
lat = np.arange(lat0, lat1, dlat)
lon = np.arange(lon0, lon1, dlon)
ny, nx = lat.size, lon.size

itp = RegularGridInterpolator((zlat, zlon), z, method="linear", bounds_error=False, fill_value=0.0)
LA, LO = np.meshgrid(lat, lon, indexing="ij")
zg = itp(np.column_stack([LA.ravel(), LO.ravel()])).reshape(ny, nx)

depth = np.where(zg < 0, -zg, 0.0)        # positive water depth
wet = zg < 0.0                             # keep flats (hold the tidal prism)
# flood fill: keep only the water body connected to the WEST (Pacific) edge
lbl, n = ndimage.label(wet)
west_labels = set(lbl[:, 0][wet[:, 0]])
keep = np.isin(lbl, list(west_labels)) & wet
wet = keep
depth = np.where(wet, np.maximum(depth, 2.0), 0.0)

dx = TARGET_M; dy = TARGET_M
print(f"grid {ny}x{nx}  dlat={dlat:.5f} dlon={dlon:.5f}  dx=dy={TARGET_M:.0f} m")
print(f"wet cells = {int(wet.sum())}  ({100*wet.mean():.1f}% of box)")
print(f"max depth = {depth.max():.1f} m")
# Golden Gate diagnostic cell
jg = int(np.argmin(np.abs(lat-37.810))); ig = int(np.argmin(np.abs(lon-(-122.478))))
print(f"Golden Gate cell (j={jg},i={ig}) depth={depth[jg,ig]:.1f} m  wet={wet[jg,ig]}")
np.savez(OUT, depth=depth, wet=wet, lat=lat, lon=lon, dx=dx, dy=dy)
print("saved", OUT)
