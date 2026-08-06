from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
import numpy as np
import pandas as pd
from skyfield.api import Loader, wgs84

ECLIPSE_DATE = dt.date(2026, 8, 12)
CENTRAL_LINE = [
    (43.35, -8.55),   # Galicja
    (42.90, -6.50),
    (42.40, -4.00),   # okolice Palencia/Burgos
    (41.90, -1.80),
    (41.55, 0.10),    # okolice Huesca/Lleida
    (41.20, 1.90),    # południe od Barcelony
    (40.60, 3.10),    # Morze Balearskie
]
HALF_WIDTH_DEG = 0.55
SEARCH_WINDOW_UTC = (
    dt.datetime(2026, 8, 12, 20, 0, 0),
    dt.datetime(2026, 8, 12, 21, 0, 0),
)
TIME_STEP_SECONDS = 1.0

@dataclass
class EclipsePointResult:
    lat: float
    lon: float
    c2_utc: dt.datetime | None
    c3_utc: dt.datetime | None
    duration_s: float
    sun_altitude_deg: float
    sun_azimuth_deg: float


def generate_grid(step_deg: float = 0.1) -> list[tuple[float, float]]:
    pts: set[tuple[float, float]] = set()
    line = np.array(CENTRAL_LINE)

    dense_lat = []
    dense_lon = []
    for i in range(len(line) - 1):
        lat0, lon0 = line[i]
        lat1, lon1 = line[i + 1]
        n = max(int(abs(lon1 - lon0) / (step_deg / 2)), 2)
        dense_lat.extend(np.linspace(lat0, lat1, n))
        dense_lon.extend(np.linspace(lon0, lon1, n))

    for lat_c, lon_c in zip(dense_lat, dense_lon):
        offsets = np.arange(-HALF_WIDTH_DEG, HALF_WIDTH_DEG + 1e-9, step_deg)
        for off in offsets:
            lat_p = round(lat_c + off, 3)
            lon_p = round(lon_c, 3)
            pts.add((lat_p, lon_p))

    return sorted(pts)



def _load_ephemeris(cache_dir: str = "./skyfield-data"):
    loader = Loader(cache_dir)
    ts = loader.timescale()
    eph = loader("de421.bsp")
    return ts, eph

def compute_point(
    lat: float,
    lon: float,
    elevation_m: float,
    ts,
    eph,
) -> EclipsePointResult:
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    observer = earth + wgs84.latlon(lat, lon, elevation_m=elevation_m)

    t0, t1 = SEARCH_WINDOW_UTC
    n_steps = int((t1 - t0).total_seconds() / TIME_STEP_SECONDS) + 1
    times_dt = [t0 + dt.timedelta(seconds=i * TIME_STEP_SECONDS) for i in range(n_steps)]
    t = ts.utc([d.year for d in times_dt], [d.month for d in times_dt],
               [d.day for d in times_dt], [d.hour for d in times_dt],
               [d.minute for d in times_dt], [d.second for d in times_dt])

    astro_sun = observer.at(t).observe(sun).apparent()
    astro_moon = observer.at(t).observe(moon).apparent()

    sun_alt, sun_az, sun_dist = astro_sun.altaz()
    moon_alt, moon_az, moon_dist = astro_moon.altaz()

    separation = astro_sun.separation_from(astro_moon).degrees

    sun_radius_deg = np.degrees(np.arcsin(696000.0 / sun_dist.km))
    moon_radius_deg = np.degrees(np.arcsin(1737.4 / moon_dist.km))
    radius_diff = moon_radius_deg - sun_radius_deg

    in_totality = (separation < radius_diff) & (sun_alt.degrees > 0)

    if not np.any(in_totality):
        return EclipsePointResult(lat, lon, None, None, 0.0, float("nan"), float("nan"))

    idx = np.where(in_totality)[0]
    c2_idx, c3_idx = idx[0], idx[-1]
    duration_s = (c3_idx - c2_idx) * TIME_STEP_SECONDS

    mid_idx = (c2_idx + c3_idx) // 2

    return EclipsePointResult(
        lat=lat,
        lon=lon,
        c2_utc=times_dt[c2_idx],
        c3_utc=times_dt[c3_idx],
        duration_s=float(duration_s),
        sun_altitude_deg=float(sun_alt.degrees[mid_idx]),
        sun_azimuth_deg=float(sun_az.degrees[mid_idx]),
    )


def build_astronomy_table(
    grid_step_deg: float = 0.1,
    elevation_lookup=None,
    cache_dir: str = "./skyfield-data",
) -> pd.DataFrame:
    ts, eph = _load_ephemeris(cache_dir)
    grid = generate_grid(grid_step_deg)

    records = []
    for lat, lon in grid:
        elevation_m = elevation_lookup(lat, lon) if elevation_lookup else 0.0
        res = compute_point(lat, lon, elevation_m, ts, eph)
        records.append(res.__dict__)

    df = pd.DataFrame.from_records(records)
    df = df[df["duration_s"] > 0].reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_astronomy_table(grid_step_deg=0.3)
    df.to_csv("astronomy_grid.csv", index=False)
