import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :]

class TrafficTransformer(nn.Module):
    """
    A Transformer-based model for traffic prediction:
    - CNN layer extracts spatial features at each time step.
    - Transformer Encoder captures complex temporal dependencies.
    """
    def __init__(self, num_nodes, input_dim, cnn_out_channels, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super(TrafficTransformer, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        
        # Spatial Layer (2D CNN) - Same logic as CNN-LSTM for consistency
        self.conv1 = nn.Conv2d(in_channels=input_dim, 
                               out_channels=cnn_out_channels, 
                               kernel_size=(3, 1), 
                               padding=(1, 0))
        
        # Project spatial features to d_model for Transformer
        self.feature_projection = nn.Linear(num_nodes * cnn_out_channels, d_model)
        
        # Temporal Layer (Transformer)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output Layer
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, input_dim = x.size()
        
        # 1. Spatial Processing
        # Reshape for CNN: (batch * seq_len, input_dim, num_nodes, 1)
        x = x.view(batch_size * seq_len, input_dim, num_nodes, 1)
        x = F.relu(self.conv1(x)) # (batch * seq_len, cnn_out, num_nodes, 1)
        
        # 2. Temporal Processing
        # Reshape for Temporal Projection: (batch, seq_len, cnn_out * num_nodes)
        x = x.view(batch_size, seq_len, -1)
        x = self.feature_projection(x) # (batch, seq_len, d_model)
        
        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # (seq_len, batch, d_model)
        
        # 3. Output
        # Take the last time step for prediction
        out = self.fc(x[-1, :, :]) # (batch, output_dim)
        
        return out

if __name__ == "__main__":
    # Test with dummy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficTransformer(num_nodes=100, input_dim=2, cnn_out_channels=8, 
                                d_model=64, nhead=4, num_layers=2, output_dim=100).to(device)
    
    # (batch, seq_len, nodes, dims)
    dummy_input = torch.randn(8, 12, 100, 2).to(device)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Transformer Model Forward Pass successful!")
