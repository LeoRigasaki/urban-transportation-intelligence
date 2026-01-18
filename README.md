# Urban Transportation Analysis

Research and implementation of transportation analytics for **Lahore** and **Riyadh** utilizing deep learning and real-time data processing.

## Project Structure

- **Lahore Traffic Monitoring**: Spatio-temporal prediction and route optimization.
- **Riyadh Transportation Analysis**: Graph-based multi-modal system integration.

## Development Progress

### Day 1: Infrastructure and Data Ingestion
- Configured Python 3.12 environment with specialized geospatial dependencies.
- Deployed PostgreSQL (PostGIS), Redis, and Kafka infrastructure via Docker.
- Extracted and processed 145,998 road nodes and 380,264 edges for the Lahore District.

### Day 2: Network Graph and Feature Engineering
- Constructed a hierarchical `networkx.MultiDiGraph` representing the Lahore road network.
- Developed a feature extraction pipeline for spatial road attributes (length, density, hierarchy).
- Validated 100% geometry integrity and topological connectivity.

### Day 3: Simulation and Deep Learning Architecture
- Implemented a CNN-LSTM hybrid model for simultaneous spatial and temporal feature learning.
- Developed a real-time traffic simulation engine using the Kafka streaming protocol.
- Resolved loss instability through feature scaling and automated temporal imputation.

#### Technical Architecture

```mermaid
graph TD
    subgraph "Data Sources"
        OSM["OpenStreetMap (OSMnx)"]
        SIM["Traffic Simulator (Python)"]
    end

    subgraph "Real-time Ingestion (Kafka)"
        T_PROD["Traffic Producer"]
        K_BUS["Kafka Topic: lahore_traffic_updates"]
        T_CONS["Traffic Consumer"]
        
        SIM --> T_PROD
        T_PROD --> K_BUS
        K_BUS --> T_CONS
    end

    subgraph "Storage Layer"
        PG["PostgreSQL + PostGIS"]
        RD["Redis (Real-time Cache)"]
        
        OSM -->|Static Road Network| PG
        T_CONS -->|Historical Training Data| PG
        T_CONS -->|Live Traffic State| RD
    end

    subgraph "Deep Learning Core (Day 3)"
        DS["Data Scaler & Imputer"]
        CNN["CNN Layer (Spatial Correlations)"]
        LSTM["LSTM Layer (Temporal Trends)"]
        PRED["Traffic Forecast"]

        PG --> DS
        DS --> CNN
        CNN --> LSTM
        LSTM --> PRED
    end
```

#### Model Validation and Performance

| Spatial Network Coverage | Temporal Traffic Trends |
|:---:|:---:|
| ![Road Network](lahore/data/plots/road_network.png) | ![Temporal Speed](lahore/data/plots/temporal_speed.png) |
| *Graph-based spatial distribution of road segments.* | *Mean speed variance across the simulation sequence.* |

| Speed Distribution | Volume Distribution |
|:---:|:---:|
| ![Speed Distribution](lahore/data/plots/speed_dist.png) | ![Volume Distribution](lahore/data/plots/volume_dist.png) |
| *Simulated speed profile (km/h).* | *Vehicle volume density per segment.* |

#### Geospatial Congestion Analysis

<p align="center">
  <img src="lahore/data/plots/congestion_static.png" width="800" alt="Congestion Map">
  <br>
  <i>Static geospatial analysis showing traffic density and road hierarchy across Lahore. Red segments indicate simulated bottlenecks.</i>
</p>

**Verification Results:**
- **Training Stability**: MSE loss reduced from 0.057 to 0.050 over initial calibration.
- **Data Throughput**: Successfully processed sequences for 20,448 nodes with multi-dimensional features.
- **Pipeline Integrity**: End-to-end verification from Kafka ingestion to model inference confirmed.

---
*Next: Day 4 - Predictive Model Optimization and Ensemble Methods*