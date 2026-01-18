# Urban Transportation Analysis

Transportation analytics platform for **Lahore** and **Riyadh** using deep learning and real-time data processing.

## 🏙️ Projects

### 🇵🇰 Lahore Traffic Monitoring
- Traffic prediction at 15/30/60 minute intervals
- Route optimization using network algorithms
- Visualization with Kepler.gl
- CNN + LSTM models

### 🇸🇦 Riyadh Transportation Analysis
- Graph Neural Networks for multi-modal systems
- Metro system integration analysis
- Pattern recognition
- Data visualization

## 🛠️ Quick Start

### Infrastructure
```bash
docker compose up -d
```

### Verification
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
source venv/bin/activate
python shared/utils/verify_infra.py
```

## 📈 Project Progress

### 📅 Day 1: Infrastructure & Data Ingestion
**Status**: ✅ Completed

**Achievements**:
- Set up **Python 3.12** environment and installed dependencies.
- Deployed **PostGIS, Redis, and Kafka** via Docker.
- Ingested **145,998 nodes** and **380,264 edges** for Lahore District.

---

### 📅 Day 2: Advanced Data Processing
**Status**: ✅ Completed

**Achievements**:
- Constructed a hierarchical **MultiDiGraph** for Lahore.
- Implemented a **Feature Engineering Pipeline** extracting spatial metrics.
- Validated **380,264 road segments** for spatial integrity.

---

### 📅 Day 3: Deep Learning Architecture
**Status**: ✅ Completed

**Achievements**:
- Implemented **CNN-LSTM** hybrid architecture for spatio-temporal forecasting.
- Built a **Traffic Simulation Pipeline** using Kafka for real-time data streaming.
- **Resolved NaN Loss Issue**: Fixed training instability through **StandardScaler** normalization and automated missing value imputation (ffill/bfill).
- Established a **Training Framework** connected to PostgreSQL for model optimization.

**Commands Executed**:
```bash
# 1. Start Traffic Simulation (Background)
python lahore/src/data_pipeline/traffic_simulator.py &
python lahore/src/data_pipeline/traffic_consumer.py &

# 2. Run Model Training
python lahore/src/ml_models/train.py
```

**Verification Results**:
- **Loss Stability**: Successfully reduced MSE loss from **0.057 → 0.050** over 10 epochs.
- **Data Integrity**: Processed **20,448 nodes** with 2 features (speed, volume) per time step.
- **Visual Validation**: Generated 4 diagnostic plots in `lahore/data/plots/`:
  - `road_network.png`: Spatial coverage across Lahore.
  - `speed_dist.png` & `volume_dist.png`: Statistical distribution of simulated traffic.
  - `temporal_speed.png`: Temporal variance capturing simulated peaks.
- **Model Health**: Forward pass and gradient descent verified.

---
*Next Up: Day 4 - Predictive Model Optimization*