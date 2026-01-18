import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedEnsemble(nn.Module):
    """
    Combines predictions from multiple models (e.g., CNN-LSTM and Transformer)
    using a gated fusion mechanism to weigh their contributions dynamically.
    """
    def __init__(self, models, output_dim):
        super(GatedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        # Gating network to predict weights for each model's prediction
        # The weights sum to 1 using softmax
        self.gate = nn.Linear(output_dim * self.num_models, self.num_models)
        
    def forward(self, x):
        all_preds = []
        for model in self.models:
            all_preds.append(model(x))
        
        # Concatenate predictions for gating: (batch, num_models * output_dim)
        concat_preds = torch.cat(all_preds, dim=1)
        
        # Predict gating weights
        gate_weights = F.softmax(self.gate(concat_preds), dim=1) # (batch, num_models)
        
        # Combine predictions: (batch, output_dim)
        final_pred = 0
        for i, pred in enumerate(all_preds):
            # gate_weights[:, i].unsqueeze(1) shape: (batch, 1)
            final_pred += gate_weights[:, i].unsqueeze(1) * pred
            
        return final_pred

if __name__ == "__main__":
    # Test with dummy predictions
    batch_size = 8
    output_dim = 100
    
    # Mock models that return target output_dim
    class MockModel(nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            self.linear = nn.Linear(1, out_dim)
        def forward(self, x):
            return self.linear(torch.ones(batch_size, 1))

    models = [MockModel(output_dim), MockModel(output_dim)]
    ensemble = GatedEnsemble(models, output_dim)
    
    # Dummy input (not used by mock models but forward expects it)
    dummy_input = torch.randn(batch_size, 12, output_dim, 2)
    output = ensemble(dummy_input)
    print(f"Ensemble output shape: {output.shape}")
    print("✅ Gated Ensemble Forward Pass successful!")
