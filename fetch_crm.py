"""One-time fetch of the NOAA Coastal Relief Model (Vol.7, 3 arc-second) subset over
San Francisco Bay via OPeNDAP -> _dev/sfbay_fine/crm_raw.npz.
Run: uv run --with netcdf4 --with numpy python3 fetch_crm.py"""
import os, numpy as np
from netCDF4 import Dataset
OUT="sfbay_fine"
os.makedirs(OUT, exist_ok=True)
URL="https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol7.nc"   # lon[-128,-117], lat[37,44]
# SF Bay box lon -122.70..-121.95, lat 37.40..38.25 -> 3 arc-sec indices:
y0,y1,x0,x1 = 480,1501,6360,7261
d=Dataset(URL)
lat=np.array(d.variables["y"][y0:y1]); lon=np.array(d.variables["x"][x0:x1])
z=np.array(d.variables["z"][y0:y1, x0:x1])     # elevation (m), negative = below MSL
print(f"lat {lat.min():.3f}..{lat.max():.3f} ({lat.size})  lon {lon.min():.3f}..{lon.max():.3f} ({lon.size})")
print(f"z range {float(z.min()):.1f}..{float(z.max()):.1f} m ; water fraction {float((z<0).mean()):.2f}")
np.savez(f"{OUT}/crm_raw.npz", lat=lat, lon=lon, z=z)
print("saved", f"{OUT}/crm_raw.npz")
