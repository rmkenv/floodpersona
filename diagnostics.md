=======================================================
DIAGNOSTIC 1 — Feature Distributions
=======================================================
       flashiness_index  time_to_peak_hr  recession_rate  peak_duration_hr  cv_discharge  seasonal_variability  base_flow_index  mean_discharge   n_events
count          242.0000         242.0000        242.0000          242.0000      242.0000              242.0000         242.0000        242.0000   242.0000
mean             0.0175          14.7581          0.1615           35.7823        1.9237              697.2839           0.6289        961.7322   212.1074
std              0.0340          59.2663          0.1317          129.5666        2.0021             2912.9656           0.1960       4531.9031   489.5942
min              0.0000           0.0333          0.0007            0.0667        0.0339                0.0000           0.0000          0.0299     1.0000
25%              0.0054           0.2500          0.0741            0.5000        1.0718               15.1810           0.5076         31.1001    35.0000
50%              0.0096           2.2500          0.1316            3.6250        1.4600               71.1053           0.6613        126.7169    70.5000
75%              0.0179           8.6875          0.2038           22.2188        2.2785              310.0691           0.7644        491.2284   157.5000
max              0.3208         625.1250          1.0000         1505.0000       25.9923            39277.9937           0.9972      65317.2651  5121.0000

  Flashiness > 0.05  (Flasher candidate): 14 gauges
  Time-to-peak < 6hr (Flasher candidate): 167 gauges
  BFI > 0.6          (Stable Baseflow)  : 149 gauges
  Time-to-peak > 48hr (Slow Riser)       : 12 gauges
  Peak duration > 200hr (Holder)         : 9 gauges

=======================================================
DIAGNOSTIC 2 — Silhouette Score Sweep (k=2..10)
=======================================================
  PCA: 6 components, 95.7% variance

  k= 2: silhouette=0.4050 ← best
  k= 3: silhouette=0.4203 ← best
  k= 4: silhouette=0.4665 ← best
  k= 5: silhouette=0.4361
  k= 6: silhouette=0.4184
  k= 7: silhouette=0.2733
  k= 8: silhouette=0.2771
  k= 9: silhouette=0.2991
  k=10: silhouette=0.3015

  → Using k=4

=======================================================
DIAGNOSTIC 3 — Cluster Feature Medians
=======================================================
         flashiness_index  time_to_peak_hr  recession_rate  peak_duration_hr  cv_discharge  seasonal_variability  base_flow_index  mean_discharge  n_events
cluster                                                                                                                                                    
0                  0.0076             2.25          0.1106              2.25        1.2909               81.8610           0.6919        132.9420      69.0
1                  0.0230             2.25          0.3116              6.25        3.2184               30.8049           0.3456         68.6344      76.0
2                  0.0037           603.25          0.0586           1064.00        2.5469               20.0811           0.8221         10.1244       4.0
3                  0.0024             1.25          0.0564             12.00        0.6776            39277.9937           0.8526      65317.2651      43.0

✓ Cluster assignments:
  Stable Baseflow: 185 gauges
  Slow Risers: 53 gauges
  Holders: 3 gauges
  Stable Baseflow: 1 gauges

✓ Reclassified GeoJSON saved: ./flood_memory/flood_memory_gauges_reclassified.geojson
✓ Reclassified CONUS map saved: ./flood_memory/conus_map_reclassified.html

═══════════════════════════════════════════
  Diagnostics Complete ✓
═══════════════════════════════════════════
  Gauges analyzed : 242
  Best k          : 4 (silhouette=0.4665)
  Output dir      : ./flood_memory

  Files:
    flood_memory_gauges_reclassified.geojson
    conus_map_reclassified.html
═══════════════════════════════════════════
