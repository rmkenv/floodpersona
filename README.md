# 🌊 Flood Memory Map

**Unsupervised machine learning pipeline that clusters USGS stream gauges by hydrological behavioral personality — how a watershed remembers and responds to rainfall.**

No API key required. All data is free and public via [USGS NWIS](https://waterservices.usgs.gov/).

---

## What It Does

Most flood analysis is event-focused. This pipeline is behavior-focused.

It pulls 2 years of 15-minute instantaneous discharge data for up to 300 active
USGS stream gauges across all 50 states, engineers 9 behavioral features per gauge,
then uses PCA + K-Means clustering to group watersheds by *how they act* — not
where they are or how big they are.

The result is a national map of **flood personalities**: distinct behavioral archetypes
that reflect the underlying hydrology, land cover, and geology of each watershed.

---

## Flood Personalities

| Personality | Behavior |
|---|---|
| 🔴 **Flashers** | Rapid response, short time-to-peak, fast recession. Urban streams, steep terrain. |
| 🟢 **Slow Risers** | Long lag time, large basin mainstems. Gradual rise and fall. |
| 🔵 **Holders** | Sustained high flows, long peak duration. Wetland or floodplain-influenced. |
| 🟡 **Stable Baseflow** | Groundwater-dominated, low variability, highly predictable. |
| 🟣 **Pulse Driven** | Snowmelt-dominated, high seasonal variability, minimal storm response. |

---

## Features Engineered

| Feature | Description |
|---|---|
| Richards-Baker Flashiness Index | Ratio of flow reversals to total flow volume |
| Time-to-Peak | Median hours from rise start to flood peak |
| Recession Rate | Rate of flow decline after peak |
| Peak Duration | Hours above 75th percentile discharge |
| CV of Discharge | Coefficient of variation — overall flow volatility |
| Seasonal Variability | Month-to-month spread in median discharge |
| Base Flow Index | Fraction of flow from groundwater (7-day min method) |
| Mean Discharge | Absolute flow scale (cfs) |
| Event Count | Number of threshold-crossing events in the record |

---

## Pipeline Architecture
