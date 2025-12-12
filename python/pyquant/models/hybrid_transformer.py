import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from ..features.gaf_transforms import GAFEncoder

class PositionalEncoding(nn.Module):
    ""
    Positional encoding for transformer models.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1)]

class TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder block with pre-normalization.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention with pre-normalization
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        
        # Feedforward with pre-normalization
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class HybridTransformerModel(nn.Module):
    """
    Hybrid Transformer model that combines GAF+CNN features with a Transformer encoder.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim: int = 3,
        use_gaf: bool = True,
        gaf_hidden_dims: list = [64, 128],
        gaf_method: str = 'summation',
        use_transformer: bool = True,
        activation: str = "gelu",
        use_pos_encoding: bool = True,
        gaf_output_ratio: float = 0.5,
    ):
        """
        Initialize the hybrid transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            output_dim: Number of output classes
            use_gaf: Whether to use GAF+CNN feature extraction
            gaf_hidden_dims: List of hidden dimensions for GAF CNN
            gaf_method: 'summation' for GASF or 'difference' for GADF
            use_transformer: Whether to use the transformer encoder
            activation: Activation function ('gelu' or 'relu')
            use_pos_encoding: Whether to use positional encoding
            gaf_output_ratio: Ratio of GAF features to raw features in the final representation
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_gaf = use_gaf
        self.use_transformer = use_transformer
        self.gaf_output_ratio = gaf_output_ratio
        
        # GAF-based feature extractor
        if use_gaf:
            gaf_output_dim = max(1, int(d_model * gaf_output_ratio))
            self.gaf_encoder = GAFEncoder(
                in_features=input_dim,
                gaf_method=gaf_method,
                hidden_dims=gaf_hidden_dims,
                output_dim=gaf_output_dim
            )
            transformer_input_dim = d_model - gaf_output_dim
        else:
            transformer_input_dim = d_model
        
        # Raw feature projection
        self.feature_proj = nn.Linear(input_dim, transformer_input_dim)
        
        # Positional encoding
        if use_pos_encoding:
            self.pos_encoder = PositionalEncoding(transformer_input_dim)
        else:
            self.pos_encoder = None
        
        # Transformer encoder
        if use_transformer:
            self.transformer_encoder = nn.ModuleList([
                TransformerEncoderBlock(
                    d_model=transformer_input_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(num_layers)
            ])
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        ""Initialize weights with Xavier/Glorot initialization.""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the hybrid transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            src_mask: Optional mask for the attention mechanism
            src_key_padding_mask: Optional mask for padding tokens
            
        Returns:
            policy: Policy logits of shape (batch_size, output_dim)
            value: Value predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Process with GAF+CNN if enabled
        gaf_features = None
        if self.use_gaf:
            # Apply GAF+CNN to the input sequence
            gaf_features = self.gaf_encoder(x)  # (batch_size, gaf_output_dim)
            
            # Project raw features
            x_proj = self.feature_proj(x)  # (batch_size, seq_len, transformer_input_dim)
            
            # Process with transformer if enabled
            if self.use_transformer:
                # Add positional encoding
                if self.pos_encoder is not None:
                    x_proj = self.pos_encoder(x_proj)
                
                # Apply transformer layers
                for layer in self.transformer_encoder:
                    x_proj = layer(
                        x_proj,
                        src_mask=src_mask,
                        src_key_padding_mask=src_key_padding_mask
                    )
                
                # Use the last time step's output
                x_transformer = x_proj[:, -1, :]  # (batch_size, transformer_input_dim)
            else:
                # If no transformer, use mean pooling
                x_transformer = x_proj.mean(dim=1)  # (batch_size, transformer_input_dim)
            
            # Combine GAF and transformer features
            combined = torch.cat([x_transformer, gaf_features], dim=1)  # (batch_size, d_model)
        else:
            # Process without GAF
            x_proj = self.feature_proj(x)  # (batch_size, seq_len, d_model)
            
            if self.use_transformer:
                # Add positional encoding
                if self.pos_encoder is not None:
                    x_proj = self.pos_encoder(x_proj)
                
                # Apply transformer layers
                for layer in self.transformer_encoder:
                    x_proj = layer(
                        x_proj,
                        src_mask=src_mask,
                        src_key_padding_mask=src_key_padding_mask
                    )
                
                # Use the last time step's output
                combined = x_proj[:, -1, :]  # (batch_size, d_model)
            else:
                # If no transformer, use mean pooling
                combined = x_proj.mean(dim=1)  # (batch_size, d_model)
        
        # Get policy and value predictions
        policy = self.policy_head(combined)  # (batch_size, output_dim)
        value = self.value_head(combined)    # (batch_size, 1)
        
        return policy, value.squeeze(-1)  # Remove last dim from value

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        ""
        Get attention weights for visualization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            attention_weights: Attention weights of shape (batch_size, nhead, seq_len, seq_len)
        """
        if not self.use_transformer:
            raise ValueError("Model does not use transformer, cannot get attention weights")
        
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x_proj = self.feature_proj(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        if self.pos_encoder is not None:
            x_proj = self.pos_encoder(x_proj)
        
        # Get attention weights from each layer
        all_attention_weights = []
        current_x = x_proj
        
        for layer in self.transformer_encoder:
            # Get self-attention module
            self_attn = layer.self_attn
            
            # Apply layer norm
            x_norm = layer.norm1(current_x)
            
            # Get query, key, value
            q = self_attn._in_proj_q(x_norm)
            k = self_attn._in_proj_k(x_norm)
            v = self_attn._in_proj_v(x_norm)
            
            # Reshape for multi-head attention
            q = q.reshape(batch_size, seq_len, self_attn.num_heads, -1).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, self_attn.num_heads, -1).transpose(1, 2)
            v = v.reshape(batch_size, seq_len, self_attn.num_heads, -1).transpose(1, 2)
            
            # Calculate attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            
            # Apply attention mask if provided
            if self_attn._in_proj_container._qkv_same_embed_dim:
                attn_scores = attn_scores.masked_fill(
                    self_attn._in_proj_container.in_proj_bias is not None,
                    float('-inf')
                )
            
            # Apply softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            all_attention_weights.append(attn_weights)
            
            # Update input for next layer
            current_x = layer(current_x)
        
        # Stack attention weights from all layers
        # Shape: (num_layers, batch_size, nhead, seq_len, seq_len)
        return torch.stack(all_attention_weights, dim=0)
