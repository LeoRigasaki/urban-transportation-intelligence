import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficCNNLSTM(nn.Module):
    """
    A hybrid architecture:
    - CNN layer to capture local spatial correlations among road segments.
    - LSTM layer to capture temporal dependencies in traffic flow.
    """
    def __init__(self, num_nodes, input_dim, cnn_out_channels, lstm_hidden_dim, output_dim):
        super(TrafficCNNLSTM, self).__init__()
        
        # Spatial Layer (1D CNN over the nodes/features)
        # Assuming input shape: (batch, seq_len, num_nodes, input_dim)
        self.conv1 = nn.Conv2d(in_channels=input_dim, 
                               out_channels=cnn_out_channels, 
                               kernel_size=(3, 1), 
                               padding=(1, 0))
        
        # Temporal Layer (LSTM)
        # Input to LSTM: (batch, seq_len, flattened_spatial_features)
        self.lstm = nn.LSTM(input_size=num_nodes * cnn_out_channels, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            dropout=0.2)
        
        # Output Layer
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, input_dim = x.size()
        
        # Reshape for CNN: (batch * seq_len, input_dim, num_nodes, 1)
        x = x.view(batch_size * seq_len, input_dim, num_nodes, 1)
        
        # CNN forward pass
        x = F.relu(self.conv1(x)) # (batch * seq_len, cnn_out, num_nodes, 1)
        
        # Reshape for LSTM: (batch, seq_len, cnn_out * num_nodes)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x) # lstm_out: (batch, seq_len, hidden_dim)
        
        # Take the last time step for prediction
        out = self.fc(lstm_out[:, -1, :])
        
        return out

    def predict_with_uncertainty(self, x, num_samples=10):
        """
        Performs Monte Carlo Dropout inference to estimate prediction uncertainty.
        """
        self.train() # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(self.forward(x).unsqueeze(0))
        
        preds = torch.cat(preds, dim=0) # (num_samples, batch, output_dim)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std

if __name__ == "__main__":
    # Test with dummy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficCNNLSTM(num_nodes=100, input_dim=5, cnn_out_channels=16, 
                           lstm_hidden_dim=64, output_dim=1).to(device)
    
    # (batch, seq_len, nodes, dims)
    dummy_input = torch.randn(8, 12, 100, 5).to(device)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Model Forward Pass successful!")
