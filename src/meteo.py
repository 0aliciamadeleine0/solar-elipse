from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from scipy.interpolate import griddata

BBOX_NORTH = 43.6
BBOX_SOUTH = 40.3
BBOX_WEST = -9.0
BBOX_EAST = 3.5

HISTORICAL_YEARS = [str(y) for y in range(2000, 2025)]
HOURS_UTC = ["17:00", "18:00", "19:00"]
CACHE_FILE = Path("era5_cloud_cover_august_evening.nc")


def chmurki(target_file: Path = CACHE_FILE) -> Path:
    import cdsapi

    if target_file.exists():
        return target_file

    client = cdsapi.Client()

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "total_cloud_cover",
            "year": HISTORICAL_YEARS,
            "month": "08",
            "day": [f"{d:02d}" for d in range(8, 17)],
            "time": HOURS_UTC,
            "area": [BBOX_NORTH, BBOX_WEST, BBOX_SOUTH, BBOX_EAST],
            "format": "netcdf",
        },
        str(target_file),
    )
    return target_file


def chmurki_ile(nc_file: Path = CACHE_FILE) -> pd.DataFrame:
    ds = xr.open_dataset(nc_file)

    cc = ds["tcc"]

    cc_mean = cc.mean(dim=[d for d in cc.dims if d != "latitude" and d != "longitude"])
    cc_std = cc.std(dim=[d for d in cc.dims if d != "latitude" and d != "longitude"])

    df = cc_mean.to_dataframe(name="cloud_cover_mean").reset_index()
    df_std = cc_std.to_dataframe(name="cloud_cover_std").reset_index()
    df = df.merge(df_std, on=["latitude", "longitude"])

    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df["p_clear"] = (1.0 - df["cloud_cover_mean"]).clip(0.0, 1.0)

    return df[["lat", "lon", "p_clear", "cloud_cover_mean", "cloud_cover_std"]]


def interpolacja(
    climatology_df: pd.DataFrame,
    target_points: pd.DataFrame,
) -> pd.DataFrame:
    from scipy.interpolate import griddata

    src_pts = climatology_df[["lon", "lat"]].values
    src_vals = climatology_df["p_clear"].values

    dst_pts = target_points[["lon", "lat"]].values
    interpolated = griddata(src_pts, src_vals, dst_pts, method="linear")

    nan_mask = np.isnan(interpolated)
    if nan_mask.any():
        nearest = griddata(src_pts, src_vals, dst_pts[nan_mask], method="nearest")
        interpolated[nan_mask] = nearest

    out = target_points.copy()
    out["p_clear"] = interpolated
    return out


if __name__ == "__main__":
    nc_path = chmurki()
    climatology = chmurki_ilr(nc_path)
    climatology.to_csv("zachmurzenie.csv", index=False)
