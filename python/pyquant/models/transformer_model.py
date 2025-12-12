import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 20,  # Number of features
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim: int = 3   # [long, neutral, short]
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.input_proj.weight.data.uniform_(-initrange, initrange)
    
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # src shape: (batch_size, seq_len, input_dim)
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer encoding
        memory = self.transformer(src, src_key_padding_mask=src_key_padding_mask)
        
        # Use the last timestep's output for prediction
        last_output = memory[:, -1, :]
        
        # Get policy and value outputs
        policy = self.policy_head(last_output)  # (batch_size, output_dim)
        value = self.value_head(last_output)    # (batch_size, 1)
        
        return policy, value.squeeze(-1)
