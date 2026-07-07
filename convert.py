#!/usr/bin/env python3
"""
Grab a real global wind snapshot from NOAA GFS and bake it into files the page
loads: wind.png (u in the red channel, v in green), wind.json (grid size, value
ranges, forecast cycle), and coastline.json (Natural Earth outlines, so you can
tell where you are). Everything here is public-domain data.

    pip install requests cfgrib ecmwflibs xarray numpy pillow
    python convert.py
"""
import os, sys, json
import datetime as dt
import numpy as np

try:
    import requests
    import xarray as xr
    from PIL import Image
except ImportError:
    sys.exit("Missing deps. Run:  pip install requests cfgrib ecmwflibs xarray numpy pillow")

W, H = 360, 180

def fetch_grib():
    # NOMADS keeps the last few 6-hourly runs; walk back until one answers.
    now = dt.datetime.now(dt.timezone.utc)
    for back in range(8):
        t = now - dt.timedelta(hours=6 * back)
        ymd, cyc = t.strftime("%Y%m%d"), (t.hour // 6) * 6
        url = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
               f"?file=gfs.t{cyc:02d}z.pgrb2.0p25.f000"
               "&lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on"
               "&leftlon=0&rightlon=360&toplat=90&bottomlat=-90"
               f"&dir=%2Fgfs.{ymd}%2F{cyc:02d}%2Fatmos")
        try:
            r = requests.get(url, timeout=90)
        except Exception as e:
            print(f"  {ymd} {cyc:02d}z {e.__class__.__name__}")
            continue
        if r.status_code == 200 and r.content[:4] == b"GRIB":
            print(f"  got {ymd} {cyc:02d}z ({len(r.content)//1024} KB)")
            return r.content, f"{ymd} {cyc:02d}Z"
        print(f"  {ymd} {cyc:02d}z not up yet (HTTP {r.status_code})")
    sys.exit("No GFS cycle answered on NOMADS. Try again in a few minutes.")

def pick(ds, *names):
    for n in names:
        if n in ds:
            return ds[n]
    sys.exit(f"none of {names} in the grib; got {list(ds.data_vars)}")

def wind():
    raw, when = fetch_grib()
    grib = "web/data/_gfs.grib2"
    with open(grib, "wb") as f:
        f.write(raw)
    ds = xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": ""})
    u = pick(ds, "u10", "10u", "ugrd10m")
    v = pick(ds, "v10", "10v", "vgrd10m")
    latn = "latitude" if "latitude" in u.coords else "lat"
    lonn = "longitude" if "longitude" in u.coords else "lon"
    lon = np.linspace(0, 360, W, endpoint=False)
    lat = np.linspace(90, -90, H)
    u = np.nan_to_num(u.interp({lonn: lon, latn: lat}).values).astype("float32")
    v = np.nan_to_num(v.interp({lonn: lon, latn: lat}).values).astype("float32")
    ds.close()
    try:
        os.remove(grib)
    except OSError:
        pass
    umin, umax, vmin, vmax = float(u.min()), float(u.max()), float(v.min()), float(v.max())
    r = np.clip((u - umin) / (umax - umin) * 255, 0, 255).round().astype("uint8")
    g = np.clip((v - vmin) / (vmax - vmin) * 255, 0, 255).round().astype("uint8")
    rgba = np.dstack([r, g, np.zeros_like(r), np.full_like(r, 255)])
    Image.fromarray(rgba, "RGBA").save("web/data/wind.png")
    json.dump({"width": W, "height": H, "uMin": umin, "uMax": umax, "vMin": vmin, "vMax": vmax,
               "source": "NOAA GFS 0.25deg 10 m wind (public domain)", "forecastTime": when},
              open("web/data/wind.json", "w"), indent=2)
    print(f"wrote wind.png + wind.json  (GFS {when})")

def coastline():
    # Natural Earth 1:110m coastline (public domain), flattened to lon/lat polylines.
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"
    try:
        gj = requests.get(url, timeout=60).json()
    except Exception as e:
        print(f"  skipped coastline ({e.__class__.__name__}); the map still runs without it")
        return
    lines = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") == "LineString":
            lines.append(geom["coordinates"])
        elif geom.get("type") == "MultiLineString":
            lines.extend(geom["coordinates"])
    lines = [[[round(x, 2), round(y, 2)] for x, y in ln] for ln in lines]
    json.dump(lines, open("web/data/coastline.json", "w"))
    print(f"wrote coastline.json ({len(lines)} lines)")

if __name__ == "__main__":
    os.makedirs("web/data", exist_ok=True)
    wind()
    coastline()
