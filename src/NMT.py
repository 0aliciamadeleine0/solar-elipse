from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import rowcol

DEM_BOUNDS = (-9.0, 40.3, 3.5, 43.6)
DEM_OUTPUT = Path("NMT.tif")

EARTH_RADIUS_M = 6_371_000.0
REFRACTION_CORRECTION_DEG = 0.57


def download_dem(bounds: tuple = DEM_BOUNDS, output: Path = DEM_OUTPUT) -> Path:
    if output.exists():
        return output

    import elevation
    
    elevation.clip(
        bounds=bounds, 
        output=str(output.absolute()), 
        max_download_tiles=100
    )
    return output


def load_dem(path: Path = DEM_OUTPUT):
    ds = rasterio.open(path)
    elevation = ds.read(1).astype(np.float32)
    return ds, elevation


def elevation_at(ds, elevation_array: np.ndarray, lat: float, lon: float) -> float:
    row, col = rowcol(ds.transform, lon, lat)
    row = int(np.clip(row, 0, elevation_array.shape[0] - 1))
    col = int(np.clip(col, 0, elevation_array.shape[1] - 1))
    val = elevation_array[row, col]
    return float(val) if val > -1000 else 0.0


def horizon_elevation_angle(
    ds,
    elevation_array: np.ndarray,
    obs_lat: float,
    obs_lon: float,
    obs_elevation_m: float,
    azimuth_deg: float,
    max_distance_km: float = 50.0,
    step_km: float = 0.5,
    observer_height_m: float = 1.7,
) -> float:
    az_rad = np.radians(azimuth_deg)
    deg_per_km_lat = 1.0 / 111.32
    obs_h = obs_elevation_m + observer_height_m

    max_angle = -90.0

    for dist_km in np.arange(step_km, max_distance_km + step_km, step_km):
        dlat = dist_km * np.cos(az_rad) * deg_per_km_lat
        dlon = dist_km * np.sin(az_rad) * deg_per_km_lat / np.cos(np.radians(obs_lat))

        pt_lat = obs_lat + dlat
        pt_lon = obs_lon + dlon

        terrain_h = elevation_at(ds, elevation_array, pt_lat, pt_lon)

        drop_m = (dist_km * 1000) ** 2 / (2 * EARTH_RADIUS_M)

        height_diff = (terrain_h - obs_h) - drop_m
        angle_deg = np.degrees(np.arctan2(height_diff, dist_km * 1000))

        max_angle = max(max_angle, angle_deg)

    return max_angle - REFRACTION_CORRECTION_DEG


def compute_horizon_clearance(
    ds,
    elevation_array: np.ndarray,
    lat: float,
    lon: float,
    obs_elevation_m: float,
    sun_altitude_deg: float,
    sun_azimuth_deg: float,
    azimuth_fan_deg: float = 15.0,
    azimuth_step_deg: float = 3.0,
) -> dict:
    azimuths = np.arange(
        sun_azimuth_deg - azimuth_fan_deg,
        sun_azimuth_deg + azimuth_fan_deg + 1e-9,
        azimuth_step_deg,
    )
    horizon_angles = [
        horizon_elevation_angle(ds, elevation_array, lat, lon, obs_elevation_m, az)
        for az in azimuths
    ]
    horizon_angle_at_sun = float(np.interp(sun_azimuth_deg, azimuths, horizon_angles))
    worst_case_angle = float(max(horizon_angles))

    margin = sun_altitude_deg - horizon_angle_at_sun
    return {
        "horizon_angle_deg": horizon_angle_at_sun,
        "horizon_worst_case_deg": worst_case_angle,
        "margin_deg": margin,
        "visible": margin > 0,
    }


if __name__ == "__main__":
    dem_path = download_dem()
    ds, elev = load_dem(dem_path)

    result = compute_horizon_clearance(
        ds, elev,
        lat=41.65, lon=-0.88, obs_elevation_m=200,
        sun_altitude_deg=12.0, sun_azimuth_deg=295.0,
    )
