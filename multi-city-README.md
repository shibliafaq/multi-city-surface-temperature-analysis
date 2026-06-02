# Multi-City Surface Temperature Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Data](https://img.shields.io/badge/Data-NASA%20MODIS%20MOD11A1-lightgrey.svg)](https://lpdaac.usgs.gov/products/mod11a1v061/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Analysis of land surface temperature (LST) across three cities at contrasting latitudes using 14 days of real NASA MODIS satellite data. Key finding: near-perfect latitude–temperature correlation (r = −0.995, R² = 0.990) across 25,905 measurements.

**[→ Live Streamlit Dashboard](https://shibliafaq-temperature-analysis.streamlit.app)**

---

## Key Findings

| City | Latitude | Mean LST | Grid cells |
|---|---|---|---|
| Dammam, Saudi Arabia | 26°N | 31.5°C | ~8,600 |
| Dublin, Ireland | 53°N | 9.6°C | ~8,600 |
| Reykjavik, Iceland | 64°N | −3.7°C | ~8,600 |

**Every 10° increase in latitude = 9.1°C decrease in mean LST** (across the 38° range sampled)

- Pearson r = −0.995 (near-perfect negative correlation)
- R² = 0.990 (latitude explains 99% of LST variance across cities)
- Total measurements: 25,905 real MODIS pixels across 14 days
- Resolution: 1 km × 1 km grid

---

## Data Source

**NASA MODIS/061/MOD11A1** — Terra Land Surface Temperature and Emissivity, Daily, 1km

Accessed via Google Earth Engine (GEE). Data collection window: 14-day period, all three cities simultaneously. Water bodies masked. Only valid LST retrievals (quality flag filtered) included.

---

## Pipeline

```
Google Earth Engine (MODIS MOD11A1)
        ↓
Python / Google Colab (data extraction + QC)
        ↓
CSV export (25,905 records)
        ↓
Streamlit dashboard + Kepler.gl 3D hexbin visualisation
```

---

## Dashboard Features

- City-by-city LST distribution plots
- Latitude vs. mean LST regression chart (r = −0.995)
- 3D hexbin map (Kepler.gl WebGL) showing spatial temperature variation
- Comparative statistics across all three cities
- Raw data export

---

## Project Structure

```
multi-city-surface-temperature-analysis/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── Data/                   # Processed MODIS data (CSV)
├── data/                   # Raw / intermediate data
└── LICENSE
```

---

## Run Locally

```bash
git clone https://github.com/shibliafaq/multi-city-surface-temperature-analysis.git
cd multi-city-surface-temperature-analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## Context

This project was completed as part of **ICS 574: Big Data Analytics** at King Fahd University of Petroleum and Minerals (KFUPM), Fall 2025.

It is one component of a broader research programme on urban heat islands across Saudi Arabian cities, which includes:
- GIS & Remote Sensing UHI Assessment of the Dammam Metropolitan Area (12,954 grid cells, manuscript in preparation)
- Smart Digital Twin Framework for UHI Monitoring & Mitigation (M.Sc. thesis, ongoing)

---

## Authors

| Name | Student ID |
|---|---|
| Shibli Afaq | G202520970 |
| Sultan Aldhafeeri | G202512390 |

Supervisor: Dr. Waleed Al-Gobi, KFUPM

---

## License

MIT — see [LICENSE](LICENSE)
