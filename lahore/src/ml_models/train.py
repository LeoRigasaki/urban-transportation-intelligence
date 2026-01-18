import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine
from shared.config.database.config import DATABASE_URL
from lahore.src.ml_models.architecture import TrafficCNNLSTM

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
        y = self.data[idx+self.seq_len, :, 0] # Predicting the first dimension (e.g. speed)
        return x, y

def train_model():
    """
    Main training loop for the Lahore traffic model.
    """
    logger.info("Starting model training pipeline...")
    
    try:
        engine = create_engine(DATABASE_URL)
        # Fetch a larger sample of historical data for training
        query = "SELECT u, v, timestamp, speed, volume FROM lahore_traffic_history ORDER BY timestamp LIMIT 50000"
        df = pd.read_sql(query, engine)
        
        if len(df) < 1000:
            logger.warning("Not enough data for meaningful training. Waiting for simulator...")
            return
        
        # Pivot data to get (time_steps, nodes, features)
        pivot_df = df.pivot_table(index='timestamp', columns=['u', 'v'], values=['speed', 'volume'])
        
        if len(pivot_df) <= 12: # seq_len is 12
            logger.warning(f"Not enough unique time steps ({len(pivot_df)}) for seq_len=12. Waiting...")
            return
            
        # 1. Handle Missing Values
        # For the smoke test, keep segments that have at least 1 data point
        threshold = 1 
        pivot_df = pivot_df.dropna(axis=1, thresh=threshold)
        logger.info(f"Filtered segments. Remaining columns: {len(pivot_df.columns)}")
        
        # Fill remaining NaNs with forward fill, then backward fill (or zero)
        pivot_df = pivot_df.ffill().bfill().fillna(0)
        
        # 2. Reshape and Scale
        # data_matrix shape: (time, nodes, features)
        # Features are 'speed' and 'volume'
        speeds = pivot_df['speed'].values
        volumes = pivot_df['volume'].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler_speed = StandardScaler()
        scaler_vol = StandardScaler()
        
        speeds_scaled = scaler_speed.fit_transform(speeds)
        volumes_scaled = scaler_vol.fit_transform(volumes)
        
        data_matrix = np.stack([speeds_scaled, volumes_scaled], axis=-1)
        num_nodes = data_matrix.shape[1]
        
        logger.info(f"Data prepared: {data_matrix.shape[0]} steps, {num_nodes} nodes, {data_matrix.shape[2]} features.")
        
        dataset = TrafficDataset(data_matrix)
        dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TrafficCNNLSTM(num_nodes=num_nodes, input_dim=2, cnn_out_channels=8, 
                               lstm_hidden_dim=32, output_dim=num_nodes).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop (demo/sample)
        model.train()
        for epoch in range(10): # Increased epochs for better test
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                if torch.isnan(loss):
                    logger.error("❌ Loss is NaN! Stopping training.")
                    return
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/10 | Loss: {total_loss/max(1, len(dataloader)):.4f}")
            
        logger.info("✅ Training sequence completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")

if __name__ == "__main__":
    train_model()
