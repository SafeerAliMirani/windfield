<div align="center">

# Windfield

**A million particles, one real NOAA wind snapshot, drawn on your GPU.**

[![Live demo](https://img.shields.io/badge/live-windfield.netlify.app-46D7FF?style=for-the-badge)](https://windfield.netlify.app)
&nbsp;
![WebGPU](https://img.shields.io/badge/WebGPU-raw%20WGSL-1f6feb?style=for-the-badge)
![Dependencies](https://img.shields.io/badge/dependencies-none-2ea043?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-8957e5?style=for-the-badge)

</div>

Windfield loads one real hour of NOAA GFS 10 m wind and pushes about a million particles through it, entirely in the browser. A hand-written WGSL compute shader advects every particle across a dark world map, and the trails glow by wind speed, from calm navy to gale gold and white. Because it is one steady snapshot, every trail is a true streamline.

## What you are looking at

- **Real data.** NOAA GFS, the US National Weather Service global forecast, 10 m wind, one 6-hourly cycle. Public domain. The forecast time sits in the panel, and it is a fixed snapshot, not a live feed.
- **A million particles on the GPU.** Positions live in a storage buffer. A WGSL compute shader samples the wind under each particle, steps it forward, and respawns the strays so the density stays even.
- **Glowing trails.** Particles draw into a floating-point screen buffer that fades a little each frame, so motion leaves light behind it. An HDR tonemap keeps the fastest winds from clipping to flat white.
- **Orientation.** Coastlines from Natural Earth and a set of major-city labels, so you always know which part of the world you are seeing.
- **Interactive.** Drag to pan, scroll to zoom into a storm, and three sliders for particle count, flow speed, and trail length.

## How it works

1. `convert.py` downloads a GRIB2 subset of 10 m U and V wind from NOAA NOMADS, regrids it to 360 by 180, and writes `web/data/wind.png` (u in the red channel, v in green) plus `web/data/wind.json` with the value ranges and the forecast cycle. It also pulls the Natural Earth coastline into `web/data/coastline.json`.
2. The page loads that texture, seeds a million particles, and runs the compute and render loop in raw WebGPU. No three.js, no framework, no build step.

## Run it

```bash
pip install requests cfgrib ecmwflibs xarray numpy pillow
python convert.py            # writes web/data/wind.png, wind.json, coastline.json
cd web && python serve.py    # then open the URL it prints
```

You need a WebGPU browser, so desktop Chrome or Edge.

## Honest notes

- It is one forecast hour, not live weather. The panel says as much.
- Wind is measured 10 m above the ground and colored by speed up to about 20 m/s.
- The map is a plain equirectangular projection centered on the Pacific.

## Author

Built by Safeer Ali Mirani ([GitHub](https://github.com/SafeerAliMirani)).

## License

MIT. NOAA GFS and Natural Earth data are public domain.
