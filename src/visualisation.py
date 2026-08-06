from __future__ import annotations
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Point


def get_spain_boundary() -> gpd.GeoDataFrame:
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    spain = world[world.name == "Spain"]
    return spain


def make_folium_heatmap(
    gdf: gpd.GeoDataFrame,
    output_html: str = "eclipse_spain_heatmap.html",
    top_n_markers: int = 10,
) -> folium.Map:
    center_lat = gdf["lat"].mean()
    center_lon = gdf["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    spain = get_spain_boundary()
    folium.GeoJson(
        spain,
        name="Granice Hiszpanii",
        style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0.0}
    ).add_to(m)

    heat_data = [
        [row["lat"], row["lon"], row["score"]]
        for _, row in gdf.iterrows()
        if row["score"] > 0
    ]
    HeatMap(
        heat_data,
        name="Ocena lokalizacji obserwacji zaćmienia Słońca w skali 1-100",
        radius=18,
        blur=20,
        max_zoom=10,
        gradient={
            "0.2": "blue",
            "0.4": "lime",
            "0.6": "yellow",
            "0.8": "orange",
            "1.0": "red",
        },
    ).add_to(m)

    if "hard_fail" in gdf.columns:
        blocked = gdf[gdf["hard_fail"]]
        blocked_layer = folium.FeatureGroup(name="Słońce zasłonięte przez teren", show=False)
        for _, row in blocked.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3,
                color="black",
                fill=True,
                fill_opacity=0.6,
                popup="Horyzont zasłania Słońce",
            ).add_to(blocked_layer)
        blocked_layer.add_to(m)

    top = gdf.sort_values("score", ascending=False).head(top_n_markers)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        popup_html = f"""
        <b>#{rank} — wynik: {row['score']:.1f}/100</b><br>
        Czas totalności: {row.get('duration_s', float('nan')):.1f} s<br>
        Margines nad horyzontem: {row.get('margin_deg', float('nan')):.2f}°<br>
        P(czyste niebo): {row.get('p_clear', float('nan')):.0%}
        """
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color="green" if rank <= 3 else "orange", icon="star"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    return m


def make_static_map(
    gdf: gpd.GeoDataFrame,
    output_png: str = "eclipse_spain_score_map.png",
    use_basemap: bool = True,
    resolution_deg: float = 0.05
):
    fig, ax = plt.subplots(figsize=(12, 10))

    spain = get_spain_boundary()
    spain.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=3)

    min_lon, min_lat, max_lon, max_lat = spain.total_bounds
    
    grid_lon, grid_lat = np.meshgrid(
        np.arange(min_lon, max_lon, resolution_deg),
        np.arange(min_lat, max_lat, resolution_deg)
    )

    points = np.column_stack((gdf["lon"], gdf["lat"]))
    scores = gdf["score"].values

    grid_scores = griddata(points, scores, (grid_lon, grid_lat), method='linear', fill_value=0.0)
    
    flat_lon = grid_lon.flatten()
    flat_lat = grid_lat.flatten()
    grid_points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(flat_lon, flat_lat)], crs="EPSG:4326")
    
    spain_geom = spain.geometry.unary_union
    mask = grid_points.within(spain_geom).values.reshape(grid_lon.shape)
    
    grid_scores[~mask] = np.nan

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=100)
    
    contour = ax.contourf(
        grid_lon, grid_lat, grid_scores, 
        levels=50, cmap=cmap, norm=norm, alpha=0.8, zorder=2
    )

    if "hard_fail" in gdf.columns:
        blocked = gdf[gdf["hard_fail"]]
        ax.scatter(blocked["lon"], blocked["lat"], c="black", s=15, marker="x",
                   label="Zasłonięte przez teren", zorder=4)

    top10 = gdf.sort_values("score", ascending=False).head(10)
    ax.scatter(top10["lon"], top10["lat"], facecolors="none",
              edgecolors="blue", s=150, linewidths=2.0, label="TOP 10", zorder=4)

    if use_basemap:
        try:
            import contextily as cx
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zorder=1)
        except Exception:
            pass

    plt.colorbar(contour, ax=ax, label="Ocena lokalizacji (0-100)")
    ax.set_xlabel("Długość geograficzna")
    ax.set_ylabel("Szerokość geograficzna")
    ax.set_title("Ocena miejsca obserwacji zaćmienia Słońca 12 sierpnia 2026 w Hiszpanii")
    ax.legend(loc="upper left")
    
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    return fig


if __name__ == "__main__":
    import pandas as pd
    from scoring import compute_scores, to_geodataframe

    demo = pd.DataFrame({
        "lat": [41.65, 41.20, 40.60, 42.90, 41.40, 40.95],
        "lon": [-0.88, 1.90, 3.10, -6.50, 0.50, 2.20],
        "duration_s": [95.0, 110.0, 60.0, 40.0, 88.0, 102.0],
        "margin_deg": [4.5, -0.5, 2.0, 1.0, 3.2, 0.8],
        "p_clear": [0.72, 0.85, 0.90, 0.55, 0.68, 0.80],
    })
    scored = compute_scores(demo)
    gdf = to_geodataframe(scored)

    make_folium_heatmap(gdf)
    make_static_map(gdf, use_basemap=False)
