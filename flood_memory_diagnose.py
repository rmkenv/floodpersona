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

PERSONALITY_NAMES  = ["Flashers", "Slow Risers", "Holders", "Stable Baseflow",
                      "Tidal Mixers", "Pulse Driven", "Rain Shadow", "Nival"]
PERSONALITY_COLORS = ["#FF4757", "#2ED573", "#1E90FF", "#FFA502",
                      "#A29BFE", "#FF6B81", "#26de81", "#fd9644"]

def assign_personality(row, idx):
    fi   = row.get("flashiness_index", 0) or 0
    bfi  = row.get("base_flow_index",  0) or 0
    ttp  = row.get("time_to_peak_hr",  0) or 0
    pd_h = row.get("peak_duration_hr", 0) or 0
    rec  = row.get("recession_rate",   0) or 0
    cv   = row.get("cv_discharge",     0) or 0

    if fi > 0.015 and ttp < 6 and rec > 0.2:  return "Flashers"
    elif bfi > 0.70 and cv < 1.5:             return "Stable Baseflow"
    elif ttp > 24:                             return "Slow Risers"
    elif pd_h > 50:                            return "Holders"
    elif cv > 2.5:                             return "Pulse Driven"
    else:                                      return PERSONALITY_NAMES[idx % len(PERSONALITY_NAMES)]

name_map, used = {}, set()
for cid, row in cluster_stats.iterrows():
    name = assign_personality(row, cid)
    if name in used:
        name = PERSONALITY_NAMES[cid % len(PERSONALITY_NAMES)]
    used.add(name)
    name_map[cid] = name

color_map = {name: PERSONALITY_COLORS[i % len(PERSONALITY_COLORS)]
             for i, name in enumerate(name_map.values())}

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
