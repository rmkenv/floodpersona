"""
flood_memory_pipeline.py
Flood Memory Map — USGS Behavioral Clustering Pipeline

Usage:
    python flood_memory_pipeline.py [--gauges 300] [--days 730] [--clusters 0] [--output ./flood_memory]

No API key required. All data is free and public (USGS NWIS).
Outputs: GeoJSON, metadata JSON, Plotly HTML maps, Folium HTML map.

Install deps first:
    pip install pandas numpy scikit-learn umap-learn requests tqdm folium plotly
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — via argparse (terminal) or edit defaults below
# ══════════════════════════════════════════════════════════════════════════════

import argparse, os

def _parse_args():
    p = argparse.ArgumentParser(description="Flood Memory Map — USGS Behavioral Clustering Pipeline")
    p.add_argument("--gauges",   type=int,   default=300,              help="Max gauges to process (default: 300)")
    p.add_argument("--days",     type=int,   default=730,              help="Days of discharge history (default: 730)")
    p.add_argument("--clusters", type=int,   default=0,                help="K-Means clusters; 0=auto via silhouette (default: 0)")
    p.add_argument("--output",   type=str,   default="./flood_memory", help="Output directory (default: ./flood_memory)")
    p.add_argument("--seed",     type=int,   default=42,               help="Random seed (default: 42)")
    return p.parse_args()

args = _parse_args()
MAX_GAUGES  = args.gauges
DAYS_HISTORY = args.days
N_CLUSTERS  = args.clusters
OUTPUT_DIR  = args.output
RANDOM_SEED = args.seed

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Config | gauges={MAX_GAUGES} | days={DAYS_HISTORY} | clusters={'auto' if N_CLUSTERS==0 else N_CLUSTERS} | output={OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
import json
import time
import pickle
from io import StringIO
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import folium

try:
    import umap
    HAS_UMAP = True
    print("✓ UMAP available")
except ImportError:
    HAS_UMAP = False
    print("⚠ umap-learn not found — will use PCA 2D for scatter view")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs): return it  # no-op fallback

print("✓ All imports loaded")

# ══════════════════════════════════════════════════════════════════════════════
# USGS GAUGE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

# US states + DC for stratified sampling
US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
]

NWIS_BASE = "https://waterservices.usgs.gov/nwis"


def fetch_gauges_for_state(state_code: str, per_state: int = 10) -> list[dict]:
    """
    Fetch active stream gauges for a state using USGS NWIS Site Service.
    Uses outputDataTypeCd=iv (not hasDataTypeCd) to avoid silent empty responses.
    """
    url = f"{NWIS_BASE}/site/"
    params = {
        "format": "rdb",
        "stateCd": state_code,
        "outputDataTypeCd": "iv",   # ← CRITICAL: hasDataTypeCd returns empty silently
        "parameterCd": "00060",     # discharge
        "siteStatus": "active",
        "siteType": "ST",           # streams only
        "hasDataTypeCd": "iv",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return _parse_rdb_sites(r.text, state_code)[:per_state]
    except Exception:
        return []


def _parse_rdb_sites(rdb_text: str, state_code: str) -> list[dict]:
    """
    Parse USGS RDB format: skip comment lines (#), skip format-code row (5s, 15s…),
    return list of site dicts. Deduplicates by site_no.
    """
    lines = [l for l in rdb_text.splitlines() if not l.startswith("#") and l.strip()]
    if len(lines) < 3:
        return []

    # Row 0 = header, Row 1 = format codes (e.g. "5s\t15s\t…"), Row 2+ = data
    header = lines[0].split("\t")
    data_lines = lines[2:]  # skip format-code row

    seen = set()
    sites = []
    for line in data_lines:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))

        site_no = row.get("site_no", "").strip()
        if not site_no or site_no in seen:
            continue
        seen.add(site_no)

        try:
            lat = float(row.get("dec_lat_va", "").strip())
            lon = float(row.get("dec_long_va", "").strip())
        except ValueError:
            continue

        sites.append({
            "site_no":    site_no,
            "station_nm": row.get("station_nm", "").strip(),
            "state":      state_code,   # we supply this — avoids KeyError: 'state'
            "lat":        lat,
            "lon":        lon,
        })
    return sites


def fetch_active_gauges(max_gauges: int = 300) -> pd.DataFrame:
    """
    Stratified sample of active USGS stream gauges across all US states.
    """
    per_state = max(1, (max_gauges // len(US_STATES)) + 2)
    all_sites = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_gauges_for_state, st, per_state): st for st in US_STATES}
        for fut in tqdm(as_completed(futures), total=len(US_STATES), desc="Discovering gauges"):
            result = fut.result()
            all_sites.extend(result)

    df = pd.DataFrame(all_sites).drop_duplicates("site_no").head(max_gauges)
    print(f"✓ Found {len(df)} active gauges across {df['state'].nunique()} states")
    return df.reset_index(drop=True)


print("Functions defined — fetching gauges next...\n")

# ══════════════════════════════════════════════════════════════════════════════
# DISCHARGE TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def fetch_discharge_series(site_no: str, days: int = 730) -> pd.Series:
    """
    Pull instantaneous discharge (param 00060) for a gauge via USGS IV service.
    Returns a DatetimeIndex Series of float discharge values (cfs).
    Returns empty Series on any failure.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    url = f"{NWIS_BASE}/iv/"
    params = {
        "format": "rdb",
        "sites": site_no,
        "parameterCd": "00060",
        "startDT": start_dt.strftime("%Y-%m-%d"),
        "endDT": end_dt.strftime("%Y-%m-%d"),
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return pd.Series(dtype=float)
        return _parse_rdb_iv(r.text)
    except Exception:
        return pd.Series(dtype=float)


def _parse_rdb_iv(rdb_text: str) -> pd.Series:
    """
    Parse USGS IV RDB response into a float discharge Series.

    USGS column naming varies by response:
      - Old style: agency_cd, site_no, datetime, tz_cd, USGS_XXXXXXXX_00060_00000, ..._cd
      - New style: agency_cd, site_no, datetime, tz_cd, XXXXXXXXX_00060_00000, ..._cd
      - Sometimes just: ..._00060, ..._00060_cd

    Strategy: find the column containing '00060' that does NOT end in '_cd',
    preferring the one that also contains '00000' (instantaneous value code).
    Falls back to pandas read_csv for robustness.
    """
    lines = rdb_text.splitlines()

    # Separate comment lines from data
    data_lines = [l for l in lines if not l.startswith("#") and l.strip()]
    if len(data_lines) < 3:
        return pd.Series(dtype=float)

    header = data_lines[0].split("\t")
    # data_lines[1] is the format-code row (e.g. "5s\t15s\t20d\t…") — skip it
    records_lines = data_lines[2:]

    # ── Locate discharge value column ─────────────────────────────────────────
    # Priority 1: contains '00060' + '00000' and does NOT end with '_cd'
    discharge_col = None
    for col in header:
        if "00060" in col and "00000" in col and not col.endswith("_cd"):
            discharge_col = col
            break
    # Priority 2: contains '00060', does not end with '_cd'
    if discharge_col is None:
        for col in header:
            if "00060" in col and not col.endswith("_cd"):
                discharge_col = col
                break
    if discharge_col is None:
        return pd.Series(dtype=float)

    # ── Locate datetime column ────────────────────────────────────────────────
    dt_col = None
    for col in header:
        if col.lower() == "datetime":
            dt_col = col
            break
    if dt_col is None and len(header) > 2:
        dt_col = header[2]  # fallback: 3rd column is always datetime in NWIS RDB
    if dt_col is None:
        return pd.Series(dtype=float)

    # ── Parse rows ────────────────────────────────────────────────────────────
    records = []
    for line in records_lines:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        raw_val = row.get(discharge_col, "").strip()
        raw_dt  = row.get(dt_col, "").strip()
        if not raw_val or not raw_dt or raw_val in ("", "Ice", "Eqp", "***", "Bkw", "Rat"):
            continue
        try:
            dt  = pd.to_datetime(raw_dt)
            val = float(raw_val)
            if val >= 0:
                records.append((dt, val))
        except (ValueError, TypeError):
            continue

    if not records:
        return pd.Series(dtype=float)

    idx, vals = zip(*records)
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()

print("Functions defined\n")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _richards_baker_flashiness(series: pd.Series) -> float:
    """
    Richards-Baker Flashiness Index: sum(|Q_t - Q_{t-1}|) / sum(Q_t).
    Higher = more flashy (urban/steep catchments). Lower = smoother (groundwater-fed).
    """
    if len(series) < 2:
        return np.nan
    diffs = series.diff().abs().dropna()
    total = series.sum()
    return float(diffs.sum() / total) if total > 0 else np.nan


def _time_to_peak(series: pd.Series) -> float:
    """
    Mean hours from start of rise to peak across detected events.
    Rise start = local minimum before a peak exceeding the 75th percentile.
    """
    if len(series) < 48:
        return np.nan
    threshold = series.quantile(0.75)
    peaks = []
    in_event = False
    rise_start = None
    peak_val = -np.inf
    peak_idx = None

    for i in range(1, len(series)):
        q = series.iloc[i]
        if not in_event and q > threshold:
            in_event = True
            rise_start = series.index[i - 1]
            peak_val = q
            peak_idx = series.index[i]
        elif in_event:
            if q > peak_val:
                peak_val = q
                peak_idx = series.index[i]
            elif q < threshold:
                if peak_idx and rise_start:
                    hrs = (peak_idx - rise_start).total_seconds() / 3600
                    peaks.append(hrs)
                in_event = False
                peak_val = -np.inf

    return float(np.median(peaks)) if peaks else np.nan


def _recession_rate(series: pd.Series) -> float:
    """
    Mean daily recession rate post-peak: average of (Q_peak - Q_24h) / Q_peak.
    """
    if len(series) < 48:
        return np.nan
    # Resample to daily
    daily = series.resample("D").mean().dropna()
    if len(daily) < 5:
        return np.nan
    threshold = daily.quantile(0.75)
    rates = []
    for i in range(len(daily) - 1):
        if daily.iloc[i] > threshold and daily.iloc[i + 1] < daily.iloc[i]:
            rate = (daily.iloc[i] - daily.iloc[i + 1]) / daily.iloc[i]
            rates.append(rate)
    return float(np.median(rates)) if rates else np.nan


def _peak_duration(series: pd.Series) -> float:
    """Hours above the 75th percentile discharge (median event length)."""
    if len(series) < 48:
        return np.nan
    threshold = series.quantile(0.75)
    above = (series > threshold).astype(int)
    durations = []
    count = 0
    freq_hrs = (series.index[1] - series.index[0]).total_seconds() / 3600 if len(series) > 1 else 0.25
    for v in above:
        if v:
            count += 1
        elif count > 0:
            durations.append(count * freq_hrs)
            count = 0
    return float(np.median(durations)) if durations else np.nan


def _base_flow_index(series: pd.Series) -> float:
    """
    Fraction of total flow from baseflow (7-day minimum method).
    Higher = more groundwater-dominated.
    """
    if len(series) < 48:
        return np.nan
    daily = series.resample("D").mean().dropna()
    if len(daily) < 14:
        return np.nan
    window = 7
    baseflow = daily.rolling(window, center=True, min_periods=1).min()
    total = daily.sum()
    return float(baseflow.sum() / total) if total > 0 else np.nan


def engineer_features(site_no: str, series: pd.Series) -> dict | None:
    """
    Engineer all 9 behavioral features for a gauge.
    Returns None if series is too short (<180 days of data).
    """
    if len(series) < 720:  # ~30 days of 15-min data
        return None

    threshold = series.quantile(0.75)
    events = []
    in_event = False
    for v in series:
        if v > threshold:
            if not in_event:
                in_event = True
                events.append(1)
        else:
            in_event = False

    monthly_medians = series.resample("ME").median().dropna()

    return {
        "site_no":             site_no,
        "flashiness_index":    _richards_baker_flashiness(series),
        "time_to_peak_hr":     _time_to_peak(series),
        "recession_rate":      _recession_rate(series),
        "peak_duration_hr":    _peak_duration(series),
        "cv_discharge":        float(series.std() / series.mean()) if series.mean() > 0 else np.nan,
        "seasonal_variability":float(monthly_medians.std()) if len(monthly_medians) > 2 else np.nan,
        "base_flow_index":     _base_flow_index(series),
        "mean_discharge":      float(series.mean()),
        "n_events":            len(events),
    }

print("Feature engineering functions defined\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — FETCH + ENGINEER
# ══════════════════════════════════════════════════════════════════════════════

# Step 1: Discover gauges
print("=" * 60)
print("FLOOD MEMORY MAP — Pipeline")
print("=" * 60)
print(f"\n[1/4] Discovering gauges (max={MAX_GAUGES})...")
gauges_df = fetch_active_gauges(MAX_GAUGES)

# Step 2: Fetch series + engineer features
print(f"\n[2/4] Fetching discharge + engineering features...")
feature_rows = []
failed = 0

def _process_gauge(row):
    series = fetch_discharge_series(row["site_no"], days=DAYS_HISTORY)
    feats = engineer_features(row["site_no"], series)
    return feats

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {
        pool.submit(_process_gauge, row): row["site_no"]
        for _, row in gauges_df.iterrows()
    }
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing gauges"):
        result = fut.result()
        if result:
            feature_rows.append(result)
        else:
            failed += 1

FEATURE_COLS = [
    "flashiness_index", "time_to_peak_hr", "recession_rate",
    "peak_duration_hr", "cv_discharge", "seasonal_variability",
    "base_flow_index", "mean_discharge", "n_events",
]

features_df = pd.DataFrame(feature_rows) if feature_rows else pd.DataFrame(columns=["site_no"] + FEATURE_COLS)
print(f"\n✓ {len(features_df)} gauges with valid features ({failed} skipped — insufficient data)")
if len(features_df) == 0:
    print("\n⚠  0 valid gauges — likely a USGS API parsing issue.")
    print("   Fetching a single gauge in debug mode to inspect the raw response...\n")
    test_site = gauges_df["site_no"].iloc[0]
    import urllib.parse
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=30)
    test_url = (f"{NWIS_BASE}/iv/?format=rdb&sites={test_site}&parameterCd=00060"
                f"&startDT={start_dt.strftime('%Y-%m-%d')}&endDT={end_dt.strftime('%Y-%m-%d')}")
    print(f"   URL: {test_url}")
    r = requests.get(test_url, timeout=30)
    lines = [l for l in r.text.splitlines() if not l.startswith("#") and l.strip()]
    print(f"   Status: {r.status_code} | Non-comment lines: {len(lines)}")
    if lines:
        print(f"   Header: {lines[0]}")
        if len(lines) > 1: print(f"   Format: {lines[1]}")
        if len(lines) > 2: print(f"   Row 1:  {lines[2][:120]}")
    raise SystemExit("Fix the parser based on the debug output above, then re-run.")
print(f"  Features: {list(features_df.columns[1:])}")
features_df.head(3)

# ══════════════════════════════════════════════════════════════════════════════
# PCA + UMAP + K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

print("[3/4] Running clustering pipeline...")

# Only dropna on columns that actually exist
existing_feature_cols = [c for c in FEATURE_COLS if c in features_df.columns]
df_clean = features_df.dropna(subset=existing_feature_cols, thresh=6).copy()
print(f"  Gauges after NaN filter: {len(df_clean)}")

# Impute remaining NaNs with column median
for col in FEATURE_COLS:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Scale
X = df_clean[FEATURE_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (retain 95% variance)
pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)
n_components = X_pca.shape[1]
print(f"  PCA: {n_components} components explain {pca.explained_variance_ratio_.sum():.1%} variance")

# UMAP for 2D scatter
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    X_2d = reducer.fit_transform(X_pca)
    embed_label = "UMAP"
else:
    pca2 = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca2.fit_transform(X_scaled)
    embed_label = "PCA"
print(f"  {embed_label} 2D embedding done")

# Silhouette-optimized K-Means (or use manual N_CLUSTERS)
if N_CLUSTERS == 0:
    print("  Auto-selecting cluster count via silhouette score...")
    best_k, best_score, best_labels = 5, -1, None
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        print(f"    k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    print(f"  → Optimal k={best_k} (silhouette={best_score:.4f})")
    cluster_labels = best_labels
    k = best_k
else:
    k = N_CLUSTERS
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = km.fit_predict(X_pca)
    print(f"  Using manual k={k}")

df_clean = df_clean.copy()
df_clean["cluster"] = cluster_labels
df_clean["umap_x"] = X_2d[:, 0]
df_clean["umap_y"] = X_2d[:, 1]

# ── Assign cluster names based on dominant behavioral signature ───────────────
PERSONALITY_NAMES = ["Flashers", "Slow Risers", "Holders", "Stable Baseflow", "Tidal Mixers",
                     "Pulse Driven", "Rain Shadow", "Nival"]
PERSONALITY_COLORS = ["#FF4757", "#2ED573", "#1E90FF", "#FFA502", "#A29BFE",
                      "#FF6B81", "#26de81", "#fd9644"]

cluster_stats = df_clean.groupby("cluster")[FEATURE_COLS].median()

def _assign_personality(stats_row, idx):
    """Heuristic: rank by flashiness, base flow, time to peak."""
    fi = stats_row.get("flashiness_index", 0) or 0
    bfi = stats_row.get("base_flow_index", 0) or 0
    ttp = stats_row.get("time_to_peak_hr", 0) or 0
    pd_h = stats_row.get("peak_duration_hr", 0) or 0

    if fi > 0.05 and ttp < 6:
        return "Flashers"
    elif bfi > 0.6:
        return "Stable Baseflow"
    elif ttp > 48:
        return "Slow Risers"
    elif pd_h > 200:
        return "Holders"
    else:
        return PERSONALITY_NAMES[idx % len(PERSONALITY_NAMES)]

name_map = {}
used_names = set()
for cid, row in cluster_stats.iterrows():
    name = _assign_personality(row, cid)
    if name in used_names:
        name = PERSONALITY_NAMES[cid % len(PERSONALITY_NAMES)]
    used_names.add(name)
    name_map[cid] = name

df_clean["cluster_name"] = df_clean["cluster"].map(name_map)
color_map = {name: PERSONALITY_COLORS[i % len(PERSONALITY_COLORS)] for i, name in enumerate(name_map.values())}

print(f"\n✓ Cluster assignments:")
for cid, name in name_map.items():
    n = (df_clean["cluster"] == cid).sum()
    print(f"  {name}: {n} gauges")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE METADATA + EXPORT GEOJSON
# ══════════════════════════════════════════════════════════════════════════════

print("[4/4] Merging metadata + exporting GeoJSON...")

# Merge cluster results with gauge lat/lon/name
result_df = df_clean.merge(
    gauges_df[["site_no", "station_nm", "state", "lat", "lon"]],
    on="site_no",
    how="left",
)

# Build GeoJSON FeatureCollection
def _build_geojson(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue
        props = {
            "site_no":             str(row.get("site_no", "")),
            "station_nm":          str(row.get("station_nm", "")),
            "state":               str(row.get("state", "")),
            "cluster":             int(row.get("cluster", -1)),
            "cluster_name":        str(row.get("cluster_name", "Unknown")),
            "flashiness_index":    _safe_float(row.get("flashiness_index")),
            "time_to_peak_hr":     _safe_float(row.get("time_to_peak_hr")),
            "recession_rate":      _safe_float(row.get("recession_rate")),
            "peak_duration_hr":    _safe_float(row.get("peak_duration_hr")),
            "cv_discharge":        _safe_float(row.get("cv_discharge")),
            "seasonal_variability":_safe_float(row.get("seasonal_variability")),
            "base_flow_index":     _safe_float(row.get("base_flow_index")),
            "mean_discharge":      _safe_float(row.get("mean_discharge")),
            "n_events":            int(row.get("n_events", 0)),
            "umap_x":              _safe_float(row.get("umap_x")),
            "umap_y":              _safe_float(row.get("umap_y")),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}

def _safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) or np.isinf(f) else round(f, 6)
    except (TypeError, ValueError):
        return None

geojson = _build_geojson(result_df)
geojson_path = os.path.join(OUTPUT_DIR, "flood_memory_gauges.geojson")
with open(geojson_path, "w") as f:
    json.dump(geojson, f)

# Also save metadata
meta = {
    "generated_at":       datetime.utcnow().isoformat(),
    "n_gauges":           len(result_df),
    "n_clusters":         k,
    "days_history":       DAYS_HISTORY,
    "embedding":          embed_label,
    "pca_components":     n_components,
    "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
    "cluster_names":      name_map,
    "feature_cols":       FEATURE_COLS,
}
with open(os.path.join(OUTPUT_DIR, "flood_memory_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"✓ GeoJSON exported: {geojson_path}")
print(f"  {len(geojson['features'])} gauge features")

# ══════════════════════════════════════════════════════════════════════════════
# CONUS MAP (Plotly → saved as HTML)
# ══════════════════════════════════════════════════════════════════════════════

fig_map = px.scatter_geo(
    result_df.dropna(subset=["lat", "lon"]),
    lat="lat",
    lon="lon",
    color="cluster_name",
    color_discrete_map=color_map,
    hover_name="station_nm",
    hover_data={
        "site_no": True,
        "state": True,
        "flashiness_index": ":.4f",
        "time_to_peak_hr": ":.1f",
        "mean_discharge": ":.1f",
        "lat": False,
        "lon": False,
    },
    title="🌊 Flood Memory Map — USGS Gauge Behavioral Clusters",
    scope="usa",
    size_max=6,
)

fig_map.update_traces(marker=dict(size=6, opacity=0.85, line=dict(width=0.4, color="white")))
fig_map.update_layout(
    geo=dict(
        bgcolor="#0D1B2A",
        landcolor="#1B3A5C",
        lakecolor="#0A1520",
        subunitcolor="#2A4A6A",
        showland=True,
        showlakes=True,
        showsubunits=True,
        coastlinecolor="#2A4A6A",
    ),
    paper_bgcolor="#0D1B2A",
    plot_bgcolor="#0D1B2A",
    font=dict(color="#88A0B0"),
    title_font=dict(color="#7EC8E3", size=16),
    legend=dict(
        bgcolor="#12253A",
        bordercolor="#1B3A5C",
        font=dict(color="#E0F0FF"),
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    height=600,
)

map_html_path = os.path.join(OUTPUT_DIR, "conus_map.html")
fig_map.write_html(map_html_path)
print(f"✓ CONUS map saved: {map_html_path}")

# ══════════════════════════════════════════════════════════════════════════════
# UMAP SCATTER (Plotly → saved as HTML)
# ══════════════════════════════════════════════════════════════════════════════

fig_scatter = px.scatter(
    result_df.dropna(subset=["umap_x", "umap_y"]),
    x="umap_x",
    y="umap_y",
    color="cluster_name",
    color_discrete_map=color_map,
    hover_name="station_nm",
    hover_data={
        "site_no": True,
        "state": True,
        "flashiness_index": ":.4f",
        "time_to_peak_hr": ":.1f",
        "cluster_name": False,
        "umap_x": False,
        "umap_y": False,
    },
    title=f"🌊 Flood Personality Space ({embed_label} 2D) — Gauges Clustered by Behavior",
    labels={"umap_x": f"{embed_label} 1", "umap_y": f"{embed_label} 2"},
)

fig_scatter.update_traces(marker=dict(size=6, opacity=0.80, line=dict(width=0.3, color="white")))
fig_scatter.update_layout(
    paper_bgcolor="#0D1B2A",
    plot_bgcolor="#0D1B2A",
    font=dict(color="#88A0B0"),
    title_font=dict(color="#7EC8E3", size=14),
    xaxis=dict(gridcolor="#1B3A5C", zerolinecolor="#1B3A5C"),
    yaxis=dict(gridcolor="#1B3A5C", zerolinecolor="#1B3A5C"),
    legend=dict(bgcolor="#12253A", bordercolor="#1B3A5C", font=dict(color="#E0F0FF")),
    height=550,
)

scatter_html_path = os.path.join(OUTPUT_DIR, "behavioral_scatter.html")
fig_scatter.write_html(scatter_html_path)
print(f"✓ Behavioral scatter saved: {scatter_html_path}")

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER FINGERPRINTS (Plotly → saved as HTML)
# ══════════════════════════════════════════════════════════════════════════════

# Normalize features 0–1 for comparison
display_features = [
    "flashiness_index", "time_to_peak_hr", "recession_rate",
    "peak_duration_hr", "cv_discharge", "base_flow_index",
]
display_labels = [
    "Flashiness", "Time-to-Peak", "Recession Rate",
    "Peak Duration", "CV Discharge", "Base Flow Index",
]

cluster_means = result_df.groupby("cluster_name")[display_features].median()
cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)

fig_bar = go.Figure()
for cluster_name in cluster_norm.index:
    color = color_map.get(cluster_name, "#888888")
    fig_bar.add_trace(go.Bar(
        name=cluster_name,
        x=display_labels,
        y=cluster_norm.loc[cluster_name].values,
        marker_color=color,
        opacity=0.85,
    ))

fig_bar.update_layout(
    title="🌊 Cluster Behavioral Fingerprints (normalized medians)",
    barmode="group",
    paper_bgcolor="#0D1B2A",
    plot_bgcolor="#0D1B2A",
    font=dict(color="#88A0B0"),
    title_font=dict(color="#7EC8E3", size=14),
    xaxis=dict(gridcolor="#1B3A5C"),
    yaxis=dict(gridcolor="#1B3A5C", title="Normalized Score (0–1)"),
    legend=dict(bgcolor="#12253A", bordercolor="#1B3A5C", font=dict(color="#E0F0FF")),
    height=450,
)

bar_html_path = os.path.join(OUTPUT_DIR, "cluster_fingerprints.html")
fig_bar.write_html(bar_html_path)
print(f"✓ Cluster fingerprints saved: {bar_html_path}")

# ══════════════════════════════════════════════════════════════════════════════
# FOLIUM INTERACTIVE MAP
# ══════════════════════════════════════════════════════════════════════════════

m = folium.Map(location=[38.5, -96], zoom_start=4, tiles="CartoDB dark_matter")

for _, row in result_df.dropna(subset=["lat", "lon"]).iterrows():
    cluster_name = row.get("cluster_name", "Unknown")
    color = color_map.get(cluster_name, "#888888").lstrip("#")

    popup_html = f"""
    <div style='font-family:monospace;font-size:12px;min-width:200px'>
      <b style='color:#{color}'>{cluster_name}</b><br>
      <b>{row.get('station_nm','')}</b><br>
      <span style='color:#888'>Site #{row.get('site_no','')}</span><br>
      <hr style='margin:4px 0'>
      Flashiness: {_safe_float(row.get('flashiness_index')) or '—'}<br>
      Time-to-Peak: {_safe_float(row.get('time_to_peak_hr')) or '—'} hr<br>
      Recession Rate: {_safe_float(row.get('recession_rate')) or '—'}<br>
      Base Flow Index: {_safe_float(row.get('base_flow_index')) or '—'}<br>
      Mean Discharge: {_safe_float(row.get('mean_discharge')) or '—'} cfs<br>
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=f"#{color}",
        fill=True,
        fill_color=f"#{color}",
        fill_opacity=0.8,
        weight=0.5,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"{row.get('station_nm','')} · {cluster_name}",
    ).add_to(m)

# Legend
legend_html = """
<div style='position:fixed;bottom:30px;left:30px;z-index:9999;
     background:#0D1B2A;border:1px solid #1B3A5C;border-radius:6px;
     padding:12px;font-family:monospace;font-size:11px;color:#88A0B0'>
  <b style='color:#7EC8E3'>Flood Personalities</b><br><br>
"""
for name, color in color_map.items():
    legend_html += f"<span style='color:{color}'>●</span> {name}<br>"
legend_html += "</div>"
m.get_root().html.add_child(folium.Element(legend_html))

folium_path = os.path.join(OUTPUT_DIR, "flood_memory_map.html")
m.save(folium_path)
print(f"✓ Folium map saved: {folium_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════════════

summary = result_df.groupby("cluster_name").agg(
    n_gauges=("site_no", "count"),
    flashiness_index=("flashiness_index", "median"),
    time_to_peak_hr=("time_to_peak_hr", "median"),
    recession_rate=("recession_rate", "median"),
    base_flow_index=("base_flow_index", "median"),
    mean_discharge=("mean_discharge", "median"),
).round(4)

summary.index.name = "Cluster"
summary.columns = ["# Gauges", "Flashiness (RBI)", "Time-to-Peak (hr)",
                   "Recession Rate", "Base Flow Index", "Mean Q (cfs)"]

print("\n📊 Cluster Behavioral Summary (median values)\n")
print(summary.to_string())

print(f"""
═══════════════════════════════════════════
  FLOOD MEMORY MAP — Pipeline Complete ✓
═══════════════════════════════════════════
  Gauges processed : {len(result_df)}
  Clusters found   : {k}
  Embedding        : {embed_label}
  Output dir       : {OUTPUT_DIR}

  Files:
    flood_memory_gauges.geojson  ← load into QGIS plugin
    flood_memory_meta.json       ← pipeline metadata
    conus_map.html               ← open in browser
    behavioral_scatter.html      ← open in browser
    cluster_fingerprints.html    ← open in browser
    flood_memory_map.html        ← interactive Folium map

  Open maps:
    open {OUTPUT_DIR}/flood_memory_map.html
═══════════════════════════════════════════
""")
