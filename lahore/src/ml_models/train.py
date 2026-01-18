import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import logging
import os
import json
from sqlalchemy import create_engine
from shared.config.database.config import DATABASE_URL
from lahore.src.ml_models.architecture import TrafficCNNLSTM
from lahore.src.ml_models.transformer_model import TrafficTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficDataset(Dataset):
    def __init__(self, data, seq_len=12):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len, :, 0] # Predicting speed
        return x, y

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # Avoid division by zero for MAPE
    y_true_stable = np.where(y_true == 0, 1e-5, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_stable)) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}

def train_model(model_type="cnn_lstm", epochs=10):
    """
    Enhanced training loop with multi-model support and advanced metrics.
    """
    logger.info(f"Starting training for model type: {model_type}")
    
    try:
        engine = create_engine(DATABASE_URL)
        query = "SELECT u, v, timestamp, speed, volume FROM lahore_traffic_history ORDER BY timestamp LIMIT 50000"
        df = pd.read_sql(query, engine)
        
        if len(df) < 1000:
            logger.warning("Insufficient data.")
            return None
        
        pivot_df = df.pivot_table(index='timestamp', columns=['u', 'v'], values=['speed', 'volume'])
        pivot_df = pivot_df.ffill().bfill().fillna(0)
        
        speeds_scaled = (pivot_df['speed'].values - pivot_df['speed'].values.mean()) / (pivot_df['speed'].values.std() + 1e-5)
        volumes_scaled = (pivot_df['volume'].values - pivot_df['volume'].values.mean()) / (pivot_df['volume'].values.std() + 1e-5)
        
        data_matrix = np.stack([speeds_scaled, volumes_scaled], axis=-1)
        num_nodes = data_matrix.shape[1]
        
        dataset = TrafficDataset(data_matrix)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == "cnn_lstm":
            model = TrafficCNNLSTM(num_nodes=num_nodes, input_dim=2, cnn_out_channels=8, 
                                   lstm_hidden_dim=32, output_dim=num_nodes).to(device)
        elif model_type == "transformer":
            model = TrafficTransformer(num_nodes=num_nodes, input_dim=2, cnn_out_channels=8,
                                        d_model=64, nhead=4, num_layers=2, output_dim=num_nodes).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        metrics_history = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            all_preds, all_trues = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    val_loss += criterion(output, y).item()
                    all_preds.append(output.cpu().numpy())
                    all_trues.append(y.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            metrics = calculate_metrics(np.concatenate(all_trues), np.concatenate(all_preds))
            logger.info(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f} | MAPE: {metrics['MAPE']:.2f}%")
            
            metrics_history.append({"epoch": epoch+1, "val_loss": avg_val_loss, **metrics})

        # Save results
        results_dir = "lahore/models/results"
        os.makedirs(results_dir, exist_ok=True)
        with open(f"{results_dir}/{model_type}_metrics.json", "w") as f:
            json.dump(metrics_history, f)
            
        torch.save(model.state_dict(), f"lahore/models/trained/{model_type}.pth")
        logger.info(f"✅ Training for {model_type} completed.")
        return metrics_history[-1]

    except Exception as e:
        logger.error(f"❌ Training failed for {model_type}: {e}")
        return None

if __name__ == "__main__":
    train_model("cnn_lstm", epochs=5)
    train_model("transformer", epochs=5)
