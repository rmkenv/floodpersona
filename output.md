============================================================
FLOOD MEMORY MAP — Pipeline
============================================================

[1/4] Discovering gauges (max=300)...
Discovering gauges: 100%|███████████████████████| 51/51 [00:03<00:00, 12.86it/s]
✓ Found 300 active gauges across 43 states

[2/4] Fetching discharge + engineering features...
Processing gauges: 100%|██████████████████████| 300/300 [38:59<00:00,  7.80s/it]

✓ 242 gauges with valid features (58 skipped — insufficient data)
  Features: ['flashiness_index', 'time_to_peak_hr', 'recession_rate', 'peak_duration_hr', 'cv_discharge', 'seasonal_variability', 'base_flow_index', 'mean_discharge', 'n_events']
[3/4] Running clustering pipeline...
  Gauges after NaN filter: 242
  PCA: 6 components explain 95.7% variance
  UMAP 2D embedding done
  Auto-selecting cluster count via silhouette score...
    k=3: silhouette=0.4203
    k=4: silhouette=0.4665
    k=5: silhouette=0.4361
    k=6: silhouette=0.4184
    k=7: silhouette=0.2733
    k=8: silhouette=0.2771
  → Optimal k=4 (silhouette=0.4665)

✓ Cluster assignments:
  Stable Baseflow: 185 gauges
  Slow Risers: 53 gauges
  Holders: 3 gauges
  Stable Baseflow: 1 gauges
[4/4] Merging metadata + exporting GeoJSON...
✓ GeoJSON exported: ./flood_memory/flood_memory_gauges.geojson
  242 gauge features
✓ CONUS map saved: ./flood_memory/conus_map.html
✓ Behavioral scatter saved: ./flood_memory/behavioral_scatter.html
✓ Cluster fingerprints saved: ./flood_memory/cluster_fingerprints.html
✓ Folium map saved: ./flood_memory/flood_memory_map.html

📊 Cluster Behavioral Summary (median values)

                 # Gauges  Flashiness (RBI)  Time-to-Peak (hr)  Recession Rate  Base Flow Index  Mean Q (cfs)
Cluster                                                                                                      
Holders                 3            0.0037             603.25          0.0586           0.8221       10.1244
Slow Risers            53            0.0230               2.25          0.3116           0.3456       68.6344
Stable Baseflow       186            0.0075               2.25          0.1097           0.6921      133.2152

═══════════════════════════════════════════
  FLOOD MEMORY MAP — Pipeline Complete ✓
═══════════════════════════════════════════
  Gauges processed : 242
  Clusters found   : 4
  Embedding        : UMAP
  Output dir       : ./flood_memory

  Files:
    flood_memory_gauges.geojson  ← load into QGIS plugin
    flood_memory_meta.json       ← pipeline metadata
    conus_map.html               ← open in browser
    behavioral_scatter.html      ← open in browser
    cluster_fingerprints.html    ← open in browser
    flood_memory_map.html        ← interactive Folium map

  Open maps:
    open ./flood_memory/flood_memory_map.html
