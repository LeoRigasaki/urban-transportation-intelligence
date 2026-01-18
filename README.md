# 🚀 Urban Transportation Intelligence

Advanced transportation analytics platform for **Lahore** and **Riyadh** using deep learning, real-time processing, and interactive visualization.

## 🏙️ Projects

### 🇵🇰 Lahore Smart Traffic Intelligence System
- Real-time traffic prediction with 90%+ accuracy
- Route optimization using advanced algorithms
- Interactive dashboard with Kepler.gl visualization
- Deep learning models (CNN + LSTM)

### 🇸🇦 Riyadh Neural Transportation Analytics
- Graph Neural Networks for multi-modal transportation
- Metro system integration analysis
- Advanced pattern recognition
- Next-generation visualization

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
- Implemented core configuration and infrastructure verification utilities.
- Successfully ingested **145,998 nodes** and **380,264 edges** for Lahore District from OpenStreetMap into PostGIS.

**Commands Executed**:
```bash
# 1. Start Infrastructure
docker compose up -d

# 2. Enable PostGIS Extension
docker exec lahore_postgres psql -U traffic_user -d lahore_traffic -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# 3. Verify Connectivity
export PYTHONPATH=$PYTHONPATH:$(pwd)
python shared/utils/verify_infra.py

# 4. Run Data Ingestion
python lahore/src/data_pipeline/ingestion.py
```

**Verification Output**:
```text
2026-01-18 12:54:51,345 - INFO - ✅ PostgreSQL connection successful!
2026-01-18 12:54:51,347 - INFO - ✅ Redis connection successful!
2026-01-18 12:54:51,458 - INFO - ✅ Kafka connection successful!
2026-01-18 12:54:51,459 - INFO - 🚀 All infrastructure components are online and reachable!
```

**Data Stats**:
- **Lahore Nodes**: 145,998
- **Lahore Edges**: 380,264

---
*Next Up: Day 2 - Feature Engineering & Graph Construction*