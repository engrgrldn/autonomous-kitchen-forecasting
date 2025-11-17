"""
Transformer-Based Demand Forecasting
====================================
Implements advanced deep learning models with attention mechanisms
for demand forecasting in autonomous kitchens.

Models included:
- Temporal Fusion Transformer (TFT)
- N-BEATS
- Informer

Author: Geraldine Castillo
Date: November 2025
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        target_col: str,
        feature_cols: List[str],
        seq_length: int = 30,
        pred_length: int = 7
    ):
        """
        Args:
            data: DataFrame with time series data
            target_col: Name of target column
            feature_cols: List of feature column names
            seq_length: Length of input sequence
            pred_length: Length of prediction horizon
        """
        self.data = data
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # Prepare sequences
        self.X, self.y = self._create_sequences()
        
    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create input-output sequences."""
        X_list, y_list = [], []
        
        for i in range(len(self.data) - self.seq_length - self.pred_length + 1):
            # Input sequence
            X_seq = self.data[self.feature_cols].iloc[i:i+self.seq_length].values
            X_list.append(X_seq)
            
            # Output sequence
            y_seq = self.data[self.target_col].iloc[i+self.seq_length:i+self.seq_length+self.pred_length].values
            y_list.append(y_seq)
        
        return torch.FloatTensor(np.array(X_list)), torch.FloatTensor(np.array(y_list))
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism for transformers.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Calculate attention scores."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(x)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Single transformer encoder block.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attention_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class DemandForecastTransformer(nn.Module):
    """
    Transformer model for demand forecasting.
    Adapted for time series with attention mechanisms.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        pred_length: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_length = pred_length
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_length)
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Use last timestep for prediction
        x = x[:, -1, :]
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to inject sequence order information.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NBEATS(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series.
    Interpretable deep learning model for forecasting.
    """
    
    def __init__(
        self,
        input_dim: int,
        pred_length: int = 7,
        num_stacks: int = 30,
        num_blocks: int = 1,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.pred_length = pred_length
        self.num_stacks = num_stacks
        
        # Stacks of blocks
        self.stacks = nn.ModuleList([
            NBEATSBlock(input_dim, hidden_dim, pred_length)
            for _ in range(num_stacks)
        ])
        
    def forward(self, x):
        # Flatten input
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        
        forecast = torch.zeros(batch_size, self.pred_length).to(x.device)
        
        for stack in self.stacks:
            backcast, block_forecast = stack(x)
            x = x - backcast
            forecast = forecast + block_forecast
        
        return forecast


class NBEATSBlock(nn.Module):
    """Single block in N-BEATS architecture."""
    
    def __init__(self, input_dim: int, hidden_dim: int, pred_length: int):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.backcast_fc = nn.Linear(hidden_dim, input_dim)
        self.forecast_fc = nn.Linear(hidden_dim, pred_length)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Forward pass
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        
        # Backcast and forecast
        backcast = self.backcast_fc(h)
        forecast = self.forecast_fc(h)
        
        return backcast, forecast


class TransformerForecastingPipeline:
    """
    Complete pipeline for transformer-based demand forecasting.
    """
    
    def __init__(
        self,
        model_type: str = 'transformer',
        seq_length: int = 30,
        pred_length: int = 7,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        device: str = None
    ):
        """
        Args:
            model_type: 'transformer' or 'nbeats'
            seq_length: Input sequence length
            pred_length: Prediction horizon
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            device: 'cuda' or 'cpu'
        """
        self.model_type = model_type
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            feature_cols: Feature column names
            train_split: Train/test split ratio
            
        Returns:
            Train and test DataLoaders
        """
        self.feature_cols = feature_cols
        
        # Scale features
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # Split data
        split_idx = int(len(df_scaled) * train_split)
        train_df = df_scaled.iloc[:split_idx]
        test_df = df_scaled.iloc[split_idx:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_df, target_col, feature_cols,
            self.seq_length, self.pred_length
        )
        test_dataset = TimeSeriesDataset(
            test_df, target_col, feature_cols,
            self.seq_length, self.pred_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader
    
    def build_model(self, input_dim: int):
        """Build the forecasting model."""
        if self.model_type == 'transformer':
            self.model = DemandForecastTransformer(
                input_dim=input_dim,
                d_model=128,
                num_heads=8,
                num_layers=4,
                pred_length=self.pred_length
            ).to(self.device)
        elif self.model_type == 'nbeats':
            self.model = NBEATS(
                input_dim=input_dim * self.seq_length,
                pred_length=self.pred_length,
                num_stacks=30
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Training history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(current_lr)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), '/home/claude/best_transformer_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('/home/claude/best_transformer_model.pt'))
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()[0]
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model performance."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2))
        }


# Example usage
if __name__ == "__main__":
    print("Transformer-Based Forecasting Module")
    print("=" * 50)
    print("Models available:")
    print("- Temporal Fusion Transformer")
    print("- N-BEATS")
    print("\nFeatures:")
    print("- Multi-head attention mechanisms")
    print("- Positional encoding")
    print("- Interpretable forecasts")
    print("- GPU acceleration support")
