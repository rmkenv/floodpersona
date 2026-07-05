"""
flood_memory_diagnose.py
Runs diagnostics + re-clusters on already-processed flood_memory data.

Usage:
    python flood_memory_diagnose.py --output ./flood_memory
"""

import argparse, os, json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
import plotly.express as px
import plotly.graph_objects as go

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--output",   type=str, default="./flood_memory", help="Same output dir as main pipeline")
p.add_argument("--clusters", type=int, default=0,                help="Force k; 0=auto (default: 0)")
p.add_argument("--seed",     type=int, default=42)
args = p.parse_args()

OUTPUT_DIR  = args.output
N_CLUSTERS  = args.clusters
RANDOM_SEED = args.seed

FEATURE_COLS = [
    "flashiness_index", "time_to_peak_hr", "recession_rate",
    "peak_duration_hr", "cv_discharge", "seasonal_variability",
    "base_flow_index", "mean_discharge", "n_events",
]

# ══════════════════════════════════════════════════════════════════════════════
# LOAD EXISTING GEOJSON
# ══════════════════════════════════════════════════════════════════════════════

geojson_path = os.path.join(OUTPUT_DIR, "flood_memory_gauges.geojson")
assert os.path.exists(geojson_path), f"GeoJSON not found at {geojson_path}"

with open(geojson_path) as f:
    geojson = json.load(f)

rows = []
for feat in geojson["features"]:
    props = feat["properties"]
    lon, lat = feat["geometry"]["coordinates"]
    props["lon"] = lon
    props["lat"] = lat
    rows.append(props)

df = pd.DataFrame(rows)
print(f"✓ Loaded {len(df)} gauges from {geojson_path}")
print(f"  Columns: {list(df.columns)}\n")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1 — Feature Distribution
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("DIAGNOSTIC 1 — Feature Distributions")
print("=" * 55)
existing_cols = [c for c in FEATURE_COLS if c in df.columns]
print(df[existing_cols].describe().round(4).to_string())

print(f"\n  Flashiness > 0.05  (Flasher candidate): {(df['flashiness_index'] > 0.05).sum()} gauges")
print(f"  Time-to-peak < 6hr (Flasher candidate): {(df['time_to_peak_hr'] < 6).sum()} gauges")
print(f"  BFI > 0.6          (Stable Baseflow)  : {(df['base_flow_index'] > 0.6).sum()} gauges")
print(f"  Time-to-peak > 48hr (Slow Riser)       : {(df['time_to_peak_hr'] > 48).sum()} gauges")
print(f"  Peak duration > 200hr (Holder)         : {(df['peak_duration_hr'] > 200).sum()} gauges")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2 — Full Silhouette Sweep
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("DIAGNOSTIC 2 — Silhouette Score Sweep (k=2..10)")
print("=" * 55)

df_clean = df.dropna(subset=existing_cols, thresh=6).copy()
for col in existing_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

X = df_clean[existing_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA: {X_pca.shape[1]} components, {pca.explained_variance_ratio_.sum():.1%} variance\n")

best_k, best_score, best_labels = 3, -1, None
scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    scores[k] = score
    marker = " ← best" if score > best_score else ""
    print(f"  k={k:2d}: silhouette={score:.4f}{marker}")
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels

# Override if user forced k
if N_CLUSTERS > 0:
    print(f"\n  Forcing k={N_CLUSTERS} as requested")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    best_labels = km.fit_predict(X_pca)
    best_k = N_CLUSTERS

df_clean["cluster"] = best_labels
print(f"\n  → Using k={best_k}")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3 — Cluster Feature Medians
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("DIAGNOSTIC 3 — Cluster Feature Medians")
print("=" * 55)
cluster_stats = df_clean.groupby("cluster")[existing_cols].median().round(4)
print(cluster_stats.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# RE-ASSIGN PERSONALITY NAMES + EXPORT UPDATED GEOJSON
# ══════════════════════════════════════════════════════════════════════════════

PERSONALITY_COLORS = {
    "Flashers": "#FF4757", "Slow Risers": "#2ED573", "Holders": "#1E90FF",
    "Stable Baseflow": "#FFA502", "Pulse Driven": "#A29BFE",
}

# Only the 5 archetypes actually defined in README.md get used. The original
# code also listed "Tidal Mixers", "Rain Shadow", "Nival" as possible labels,
# but those three are never given a behavioral definition anywhere in the
# repo -- using them would mean inventing meaning that isn't documented, so
# they're excluded here rather than silently assigned.
#
# Each archetype is defined as a signed combination of z-scored features
# (z relative to the full gauge population), reflecting the behavioral
# description in README.md. Every cluster is then assigned to its best-fit
# archetype via optimal (Hungarian) linear-sum assignment on these scores,
# which finds the globally best one-to-one labeling and makes duplicate
# labels structurally impossible -- no threshold cutoffs, no fallback list,
# no collision handling needed.
ARCHETYPE_WEIGHTS = {
    # feature: weight (positive = "high value supports this archetype")
    "Flashers":         {"flashiness_index": 1.0, "recession_rate": 1.0,
                          "time_to_peak_hr": -1.0, "peak_duration_hr": -1.0},
    "Slow Risers":      {"time_to_peak_hr": 1.0, "peak_duration_hr": 0.5,
                          "flashiness_index": -1.0},
    "Holders":          {"peak_duration_hr": 1.0, "time_to_peak_hr": 0.5},
    "Stable Baseflow":  {"base_flow_index": 1.0, "cv_discharge": -1.0,
                          "flashiness_index": -1.0},
    "Pulse Driven":     {"seasonal_variability": 1.0, "flashiness_index": -1.0},
}

def compute_archetype_scores(cluster_stats, population_mean, population_std):
    """Z-score each cluster's median feature vector against the full gauge
    population, then score it against every archetype's defining weights.
    Returns a (clusters x archetypes) score matrix.
    """
    archetypes = list(ARCHETYPE_WEIGHTS.keys())
    z = (cluster_stats - population_mean) / population_std
    scores = pd.DataFrame(index=cluster_stats.index, columns=archetypes, dtype=float)
    for arch, weights in ARCHETYPE_WEIGHTS.items():
        s = sum(z[feat] * w for feat, w in weights.items() if feat in z.columns)
        scores[arch] = s
    return scores

population_mean = df_clean[existing_cols].mean()
population_std = df_clean[existing_cols].std()
archetype_scores = compute_archetype_scores(cluster_stats, population_mean, population_std)

# Hungarian assignment maximizes total match quality; linear_sum_assignment
# minimizes cost, so we negate scores. Pads with dummy rows/cols automatically
# when #clusters != #archetypes (scipy handles rectangular matrices).
cost = -archetype_scores.values
row_ind, col_ind = linear_sum_assignment(cost)
name_map = {archetype_scores.index[r]: archetype_scores.columns[c]
            for r, c in zip(row_ind, col_ind)}

color_map = {name: PERSONALITY_COLORS.get(name, "#999999") for name in name_map.values()}

df_clean["cluster_name"] = df_clean["cluster"].map(name_map)

print("\n✓ Cluster assignments:")
for cid, name in name_map.items():
    n = (df_clean["cluster"] == cid).sum()
    print(f"  {name}: {n} gauges")

# ── Merge lat/lon back and export updated GeoJSON ────────────────────────────
result_df = df_clean.merge(
    df[["site_no", "station_nm", "state", "lat", "lon"]],
    on="site_no", how="left", suffixes=("", "_orig")
)

def safe_float(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except:
        return None

features = []
for _, row in result_df.iterrows():
    if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
        continue
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
        "properties": {
            "site_no":      str(row.get("site_no", "")),
            "station_nm":   str(row.get("station_nm", "")),
            "state":        str(row.get("state", "")),
            "cluster":      int(row.get("cluster", -1)),
            "cluster_name": str(row.get("cluster_name", "Unknown")),
            **{c: safe_float(row.get(c)) for c in existing_cols},
        }
    })

out_geojson = {"type": "FeatureCollection", "features": features}
out_path = os.path.join(OUTPUT_DIR, "flood_memory_gauges_reclassified.geojson")
with open(out_path, "w") as f:
    json.dump(out_geojson, f)
print(f"\n✓ Reclassified GeoJSON saved: {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# UPDATED CONUS MAP
# ══════════════════════════════════════════════════════════════════════════════

fig = px.scatter_geo(
    result_df.dropna(subset=["lat", "lon"]),
    lat="lat", lon="lon",
    color="cluster_name",
    color_discrete_map=color_map,
    hover_name="station_nm",
    hover_data={"site_no": True, "state": True,
                "flashiness_index": ":.4f", "time_to_peak_hr": ":.1f",
                "lat": False, "lon": False},
    title=f"🌊 Flood Memory Map — Reclassified (k={best_k})",
    scope="usa",
)
fig.update_traces(marker=dict(size=6, opacity=0.85, line=dict(width=0.4, color="white")))
fig.update_layout(
    geo=dict(bgcolor="#0D1B2A", landcolor="#1B3A5C", lakecolor="#0A1520",
             showland=True, showlakes=True, showsubunits=True,
             subunitcolor="#2A4A6A", coastlinecolor="#2A4A6A"),
    paper_bgcolor="#0D1B2A", plot_bgcolor="#0D1B2A",
    font=dict(color="#88A0B0"),
    title_font=dict(color="#7EC8E3", size=16),
    legend=dict(bgcolor="#12253A", bordercolor="#1B3A5C", font=dict(color="#E0F0FF")),
    margin=dict(l=0, r=0, t=50, b=0), height=600,
)
map_path = os.path.join(OUTPUT_DIR, "conus_map_reclassified.html")
fig.write_html(map_path)
print(f"✓ Reclassified CONUS map saved: {map_path}")

print(f"""
═══════════════════════════════════════════
  Diagnostics Complete ✓
═══════════════════════════════════════════
  Gauges analyzed : {len(result_df)}
  Best k          : {best_k} (silhouette={best_score:.4f})
  Output dir      : {OUTPUT_DIR}

  Files:
    flood_memory_gauges_reclassified.geojson
    conus_map_reclassified.html
═══════════════════════════════════════════
""")
