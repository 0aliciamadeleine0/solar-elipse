from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata


@dataclass
class ScoringConfig:
    czas_trwania_zacmienia: float = 0.30
    wysokosciowe: float = 0.35
    chmurowe: float = 0.35
    horyzontowe_max: float = 3.0

    def __post_init__(self):
        total = self.czas_trwania_zacmienia + self.wysokosciowe + self.chmurowe
        assert abs(total - 1.0) < 1e-6


def score_duration(duration_s: pd.Series) -> pd.Series:
    max_duration = duration_s.max()
    if max_duration <= 0:
        return duration_s * 0.0
    return (duration_s / max_duration).clip(0.0, 1.0)


def score_horizon(margin_deg: pd.Series, horyzontowe_max: float) -> tuple[pd.Series, pd.Series]:
    hard_fail = margin_deg <= 0
    s = (margin_deg / horyzontowe_max).clip(0.0, 1.0)
    s[hard_fail] = 0.0
    return s, hard_fail


def score_weather(p_clear: pd.Series) -> pd.Series:
    return p_clear.clip(0.0, 1.0)


def compute_scores(df: pd.DataFrame, config: ScoringConfig = ScoringConfig()) -> pd.DataFrame:
    out = df.copy()

    out["s_duration"] = score_duration(out["duration_s"])
    out["s_horizon"], out["hard_fail"] = score_horizon(out["margin_deg"], config.horyzontowe_max)
    out["s_weather"] = score_weather(out["p_clear"])

    raw_score = (
        config.czas_trwania_zacmienia * out["s_duration"]
        + config.wysokosciowe * out["s_horizon"]
        + config.chmurowe * out["s_weather"]
    )

    out["score"] = np.where(out["hard_fail"], 0.0, raw_score * 100.0)

    return out.sort_values("score", ascending=False).reset_index(drop=True)


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def merge_all_sources(
    astronomy_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = astronomy_df.merge(horizon_df, on=["lat", "lon"], how="left")
    merged = merged.merge(weather_df, on=["lat", "lon"], how="left")
    merged["p_clear"] = merged["p_clear"].fillna(merged["p_clear"].median())
    return merged


def interpolate_smooth_grid(
    scored_df: pd.DataFrame, 
    resolution_deg: float = 0.02, 
    method: str = 'linear'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  
    min_lat, max_lat = scored_df["lat"].min(), scored_df["lat"].max()
    min_lon, max_lon = scored_df["lon"].min(), scored_df["lon"].max()

    grid_lon, grid_lat = np.meshgrid(
        np.arange(min_lon, max_lon, resolution_deg),
        np.arange(min_lat, max_lat, resolution_deg)
    )

    points = scored_df[["lon", "lat"]].values
    scores = scored_df["score"].values

    grid_scores = griddata(points, scores, (grid_lon, grid_lat), method=method)
    grid_scores = np.nan_to_num(grid_scores, nan=0.0)

    return grid_lon, grid_lat, grid_scores
