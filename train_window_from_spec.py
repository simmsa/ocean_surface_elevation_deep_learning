from pathlib import Path
import argparse
import math

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler


#  Data Loading ---------------------------------------------------------{{{


class WaveSpecification:
    def __init__(self, spec_file: str):
        self.spec_df = pd.read_parquet(spec_file)
        self._validate_specification()

    def _validate_specification(self):
        required_cols = ["path"]
        missing_cols = [col for col in required_cols if col not in self.spec_df.columns]
        if missing_cols:
            raise ValueError(
                f"Specification file missing required columns: {missing_cols}"
            )

    def get_data_paths(self) -> List[str]:
        paths = self.spec_df["path"].tolist()
        # Validate paths exist
        invalid_paths = [path for path in paths if not Path(path).exists()]
        if invalid_paths:
            raise FileNotFoundError(f"Following data files not found: {invalid_paths}")
        return paths


class WaveDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        input_columns: List[str],
        output_columns: List[str],
        window_size: int = 128,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
    ):
        self.window_size = window_size
        self.input_columns = input_columns
        self.output_columns = output_columns

        self.df = pd.read_parquet(data_path)
        self.data_path = data_path

        self.scaler = scaler if scaler is not None else StandardScaler()
        if fit_scaler:
            self.df[input_columns] = self.scaler.fit_transform(self.df[input_columns])
        else:
            self.df[input_columns] = self.scaler.transform(self.df[input_columns])

        # Create input-output pairs
        self.data = []
        self.labels = []
        self._create_sequences()

    def _create_sequences(self):
        """Creates input-output sequence pairs where output is the next window after input"""
        inputs = self.df[self.input_columns].values
        outputs = self.df[self.output_columns].values

        # Need enough room for both input and output windows
        for i in range(len(self.df) - 2 * self.window_size + 1):
            # Input is current window
            input_seq = inputs[i : i + self.window_size]
            # Output is next window
            output_seq = outputs[i + self.window_size : i + 2 * self.window_size]

            self.data.append(torch.tensor(input_seq, dtype=torch.float32))
            self.labels.append(torch.tensor(output_seq, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.data[idx], self.labels[idx], self.data_path


class WaveDataManager:
    def __init__(
        self,
        spec_file: str,
        input_columns: List[str],
        output_columns: List[str],
        window_size: int = 128,
        batch_size: int = 16,
        validation_split: float = 0.2,
    ):
        self.spec = WaveSpecification(spec_file)
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.window_size = window_size
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Will be set when data is loaded
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._combined_dataset = None

    def get_combined_dataset(self) -> ConcatDataset:
        if self._combined_dataset is None:
            datasets = []
            paths = self.spec.get_data_paths()

            # Create first dataset to fit scaler
            first_dataset = WaveDataset(
                paths[0],
                self.input_columns,
                self.output_columns,
                self.window_size,
                scaler=None,
                fit_scaler=True,
            )
            self.scaler = first_dataset.scaler
            datasets.append(first_dataset)

            # Create remaining datasets using the fitted scaler
            for path in paths[1:]:
                dataset = WaveDataset(
                    path,
                    self.input_columns,
                    self.output_columns,
                    self.window_size,
                    scaler=self.scaler,
                    fit_scaler=False,
                )
                datasets.append(dataset)

            self._combined_dataset = ConcatDataset(datasets)

        return self._combined_dataset

    def prepare_data_splits(
        self, test_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepares train, validation, and test dataloaders"""
        combined_dataset = self.get_combined_dataset()

        # First split into train+val and test
        total_size = len(combined_dataset)
        test_size = int(total_size * test_split)
        train_val_size = total_size - test_size

        train_val_dataset, test_dataset = random_split(
            combined_dataset,
            [train_val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Then split train_val into train and validation
        train_size = int(train_val_size * (1 - self.validation_split))
        val_size = train_val_size - train_size

        train_dataset, val_dataset = random_split(
            train_val_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            persistent_workers=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )

        return self.train_loader, self.val_loader, self.test_loader


#  End Data Loading -----------------------------------------------------}}}
#  Model Manager --------------------------------------------------------{{{


class WavePredictionModel(pl.LightningModule):
    def __init__(self, input_dim: int, learning_rate: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Initialize history storage for epochs
        self.history = {
            "train_mse": [],
            "val_mse": [],
            "train_mae": [],
            "val_mae": [],
            "train_r2": [],
            "val_r2": [],
            "learning_rates": [],
        }

        # Current epoch metrics
        self.current_train_metrics = []
        self.current_val_metrics = []
        self.current_test_metrics = []

        # Track validation runs
        self.validation_epoch_count = 0

        self.save_hyperparameters()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict:
        x, y, source_path = batch
        y_hat = self(x)

        # Calculate metrics
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        y_var = torch.var(y, unbiased=False)
        r2 = 1 - mse_loss / y_var

        # Store batch metrics
        self.current_train_metrics.append(
            {"mse": mse_loss.item(), "mae": mae_loss.item(), "r2": r2.item()}
        )

        # Log metrics
        self.log("train_loss", mse_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_mae", mae_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_r2", r2, prog_bar=True, on_epoch=True, on_step=False)

        return {"loss": mse_loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict:
        x, y, source_path = batch
        y_hat = self(x)

        # Calculate metrics
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        y_var = torch.var(y, unbiased=False)
        r2 = 1 - mse_loss / y_var

        # Store batch metrics
        self.current_val_metrics.append(
            {"mse": mse_loss.item(), "mae": mae_loss.item(), "r2": r2.item()}
        )

        # Log metrics
        self.log("val_loss", mse_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_mae", mae_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_r2", r2, prog_bar=True, on_epoch=True, on_step=False)

        return {"val_loss": mse_loss}

    def on_train_epoch_start(self):
        self.current_train_metrics = []

    def on_validation_epoch_start(self):
        self.current_val_metrics = []
        self.validation_epoch_count += 1

    def on_train_epoch_end(self):
        # Calculate epoch averages
        avg_mse = np.mean([x["mse"] for x in self.current_train_metrics])
        avg_mae = np.mean([x["mae"] for x in self.current_train_metrics])
        avg_r2 = np.mean([x["r2"] for x in self.current_train_metrics])

        # Store in history
        self.history["train_mse"].append(avg_mse)
        self.history["train_mae"].append(avg_mae)
        self.history["train_r2"].append(avg_r2)

        # Store current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.history["learning_rates"].append(current_lr)

    def on_validation_epoch_end(self):
        # Calculate epoch averages
        avg_mse = np.mean([x["mse"] for x in self.current_val_metrics])
        avg_mae = np.mean([x["mae"] for x in self.current_val_metrics])
        avg_r2 = np.mean([x["r2"] for x in self.current_val_metrics])

        # Only store validation metrics every second validation run (at end of training epoch)
        # if self.validation_epoch_count % 2 == 0:
        self.history["val_mse"].append(avg_mse)
        self.history["val_mae"].append(avg_mae)
        self.history["val_r2"].append(avg_r2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict:
        """Evaluation step during testing"""
        x, y, source_path = batch
        y_hat = self(x)

        # Calculate metrics
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        # Calculate R² score
        y_var = torch.var(y, unbiased=False)
        r2 = 1 - mse_loss / y_var

        # Store batch metrics
        self.current_test_metrics.append(
            {"mse": mse_loss.item(), "mae": mae_loss.item(), "r2": r2.item()}
        )

        # Log metrics
        self.log("test_loss", mse_loss, prog_bar=True)
        self.log("test_mae", mae_loss, prog_bar=True)
        self.log("test_r2", r2, prog_bar=True)

        return {
            "test_loss": mse_loss,
            "test_mae": mae_loss,
            "test_mse": mse_loss,
            "test_r2": r2,
        }

    def on_test_epoch_start(self):
        """Reset metrics at start of test epoch"""
        self.current_test_metrics = []

    def on_test_epoch_end(self):
        """Calculate and log final test metrics"""
        # Calculate averages
        avg_mse = np.mean([x["mse"] for x in self.current_test_metrics])
        avg_mae = np.mean([x["mae"] for x in self.current_test_metrics])
        avg_r2 = np.mean([x["r2"] for x in self.current_test_metrics])

        # Log final metrics
        self.log("test_mse_avg", avg_mse)
        self.log("test_mae_avg", avg_mae)
        self.log("test_r2_avg", avg_r2)

        # Optional: store test results in history
        if "test_metrics" not in self.history:
            self.history["test_metrics"] = {
                "mse": avg_mse,
                "mae": avg_mae,
                "r2": avg_r2,
            }


#  End Model Manager ----------------------------------------------------}}}
#  NN Models ------------------------------------------------------------{{{

#  LSTM Model -----------------------------------------------------------{{{


class LSTMModel(WavePredictionModel):
    """LSTM-based model for wave prediction"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        super().__init__(input_dim, learning_rate)
        self.save_hyperparameters()  # Save all init parameters

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions


#  End LSTM Model -------------------------------------------------------}}}
#  EnhancedLSTM --------------------------------------------------------{{{


class EnhancedLSTMModel(WavePredictionModel):
    """Enhanced LSTM model for wave prediction with layered architecture"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        bidirectional: bool = True,
    ):
        super().__init__(input_dim, learning_rate)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # Main LSTM layers with skip connections
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        lstm_input_dim = hidden_dim
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        for _ in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    lstm_input_dim,
                    hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=0,
                )
            )
            self.layer_norms.append(nn.LayerNorm(lstm_output_dim))
            lstm_input_dim = lstm_output_dim

        # Output processing
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Linear(hidden_dim // 2, input_dim),
            ]
        )

        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(len(self.output_layers))]
        )

        # Skip connection
        self.skip_connection = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original input for skip connection
        original_input = x

        # Input processing
        x = self.input_layer(x)

        # Process through LSTM layers with residual connections
        for lstm, norm in zip(self.lstm_layers, self.layer_norms):
            residual = x
            x, _ = lstm(x)
            x = norm(x)
            if residual.shape == x.shape:
                x = x + residual

        # Output processing
        for linear, dropout in zip(self.output_layers[:-1], self.dropouts[:-1]):
            residual = x
            x = linear(x)
            x = F.relu(x)
            x = dropout(x)
            if residual.shape == x.shape:
                x = x + residual

        # Final output layer
        x = self.output_layers[-1](x)

        # Add skip connection
        x = x + self.skip_connection(original_input)

        return x

    def configure_optimizers(self):
        """Simple Adam optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


#  End EnhancedLSTM ----------------------------------------------------}}}
#  Transformer Model ----------------------------------------------------{{{


class TransformerModel(WavePredictionModel):
    """Transformer-based model for wave prediction"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        super().__init__(input_dim, learning_rate)
        self.save_hyperparameters()  # Save all init parameters

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        return self.output_projection(x)


#  End Transformer Model ------------------------------------------------}}}
#  Enhanced Transformer Model -------------------------------------------{{{


class EnhancedTransformerModel(WavePredictionModel):
    """Enhanced Transformer-based model for wave prediction"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,  # Reduced from 8
        num_layers: int = 3,  # Reduced from 4
        dropout: float = 0.1,  # Reduced from 0.2
        learning_rate: float = 1e-3,
    ):
        super().__init__(input_dim, learning_rate)
        self.save_hyperparameters()

        # Positional encoding
        self.register_buffer(
            "positions", self._create_positional_encoding(128, d_model)
        )

        # Input processing
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Reduced from 4x
            dropout=dropout,
            batch_first=True,
            activation="gelu",  # Changed from ReLU
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        # Output processing
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, input_dim),
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = self.input_projection(x)
        x = x + self.positions[: x.size(1)]

        # Create attention mask for local attention
        mask = self._generate_local_mask(x.size(1)).to(x.device)

        # Apply transformer
        x = self.transformer_encoder(x, mask=mask)
        return self.output_projection(x)

    def _generate_local_mask(self, size: int, window: int = 32) -> torch.Tensor:
        """Generate mask for local attention"""
        mask = torch.ones(size, size) * float("-inf")
        for i in range(size):
            start = max(0, i - window)
            end = min(size, i + window + 1)
            mask[i, start:end] = 0
        return mask


#  End Enhanced Transformer Model ---------------------------------------}}}
#  Convolutional LSTM ---------------------------------------------------{{{


class ConvLSTMModel(WavePredictionModel):
    """Hybrid CNN-LSTM model for wave prediction"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        super().__init__(input_dim, learning_rate)
        self.save_hyperparameters()

        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding="same"),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding="same"),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,  # Use bidirectional LSTM
        )

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for CNN (batch, channels, sequence_length)
        x = x.transpose(1, 2)

        # Apply CNN
        x = self.cnn(x)

        # Reshape for LSTM (batch, sequence_length, features)
        x = x.transpose(1, 2)

        # Apply LSTM
        lstm_out, _ = self.lstm(x)

        # Apply output layers
        predictions = self.output_layer(lstm_out)

        return predictions


#  End Convolutional LSTM -----------------------------------------------}}}

#  End NN Models --------------------------------------------------------}}}
#  Trainer Class --------------------------------------------------------{{{


class WavePredictor:
    """Main class for training and evaluating wave prediction models"""

    def __init__(
        self,
        model_type: str,
        input_dim: int,
        window_size: int = 128,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        model_params: Optional[Dict] = None,
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model_params = model_params or {}

        # Initialize model
        self.model = self._create_model()

    def _create_model(self) -> WavePredictionModel:
        """Create the specified model type"""
        if self.model_type.lower() == "lstm":
            return LSTMModel(
                input_dim=self.input_dim,
                learning_rate=self.learning_rate,
                **self.model_params,
            )
        elif self.model_type.lower() == "enhanced_lstm":
            return LSTMModel(
                input_dim=self.input_dim,
                learning_rate=self.learning_rate,
                **self.model_params,
            )
        elif self.model_type.lower() == "transformer":
            return TransformerModel(
                input_dim=self.input_dim,
                learning_rate=self.learning_rate,
                **self.model_params,
            )
        elif self.model_type.lower() == "enhanced_transformer":
            return EnhancedTransformerModel(
                input_dim=self.input_dim,
                learning_rate=self.learning_rate,
                **self.model_params,
            )
        elif self.model_type.lower() == "conv_lstm":
            return ConvLSTMModel(
                input_dim=self.input_dim,
                learning_rate=self.learning_rate,
                **self.model_params,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_data(
        self, data_manager: WaveDataManager, test_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test datasets"""
        # Get full dataset
        full_dataset = data_manager.get_combined_dataset()

        # Split into train+val and test
        test_size = int(len(full_dataset) * test_split)
        train_val_size = len(full_dataset) - test_size

        train_val_dataset, test_dataset = random_split(
            full_dataset,
            [train_val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Split train+val into train and validation
        val_size = int(train_val_size * 0.2)
        train_size = train_val_size - val_size

        train_dataset, val_dataset = random_split(
            train_val_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader

    def save_training_history(self, model_label, save_dir: str = "training_history"):
        """Save training history to a parquet file with timestamp"""
        save_dir = Path(save_dir, model_label)
        save_dir.mkdir(exist_ok=True)

        # Get number of epochs completed
        n_epochs = len(self.model.history["train_mse"])

        # Create DataFrame with all metrics
        history_dict = {
            "epoch": list(range(1, n_epochs + 1)),
            "train_mse": self.model.history["train_mse"],
            "val_mse": self.model.history["val_mse"][1:],
            "train_mae": self.model.history["train_mae"],
            "val_mae": self.model.history["val_mae"][1:],
            "train_r2": self.model.history["train_r2"],
            "val_r2": self.model.history["val_r2"][1:],
            "learning_rate": self.model.history["learning_rates"],
        }

        for key in history_dict.keys():
            print(f"{key}: len({len(history_dict[key])})")
            print(f"{key}: {history_dict[key]})")

        # Create DataFrame
        history_df = pd.DataFrame(history_dict)

        # Add metadata columns
        history_df["model_type"] = self.model_type
        history_df["input_dim"] = self.input_dim
        history_df["window_size"] = self.window_size
        history_df["batch_size"] = self.batch_size
        history_df["initial_learning_rate"] = self.learning_rate
        history_df["max_epochs"] = self.max_epochs

        # Add model parameters
        for param_name, param_value in self.model_params.items():
            history_df[f"param_{param_name}"] = param_value

        # Generate timestamp for unique filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        file_path = save_dir / f"training_history_{model_label}_{timestamp}.parquet"

        # Save to parquet
        history_df.to_parquet(file_path)
        print(f"Training history saved to: {file_path}")

        return history_df

    # Update WavePredictor class to include saving history after training
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_label,
        save_dir: str = "models",
        save_history: bool = True,
        history_dir: str = "training_history",
    ) -> str:
        """Train the model and optionally save training history"""
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename=f"wave_predictor_{self.model_type}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        # early_stop_callback = EarlyStopping(
        #     monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min"
        # )

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            # callbacks=[checkpoint_callback, early_stop_callback],
            callbacks=[checkpoint_callback],
            accelerator="auto",
            devices=1,
        )

        # Train model
        trainer.fit(self.model, train_loader, val_loader)

        # Save training history if requested
        if save_history:
            self.save_training_history(model_label, history_dir)

        return checkpoint_callback.best_model_path

    def evaluate(
        self, test_loader: DataLoader, model_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate the model on test data"""
        if model_path:
            # Load the checkpoint with the saved hyperparameters
            if self.model_type.lower() == "lstm":
                self.model = LSTMModel.load_from_checkpoint(model_path, strict=True)
            elif self.model_type.lower() == "transformer":
                self.model = TransformerModel.load_from_checkpoint(
                    model_path, strict=True
                )

        trainer = pl.Trainer(accelerator="auto", devices=1)
        test_results = trainer.test(self.model, test_loader)

        print(self.model.history)

        return test_results[0]

    def save_predictions(
        self,
        dataloader: DataLoader,
        output_column_names,
        folder,
        num_samples: int = 3,
        save_path: Optional[str] = None,
        save_dir="testing_history",
    ):
        """Visualize input sequence and predictions vs actual next sequence"""
        self.model.eval()
        all_inputs = []
        all_predictions = []
        all_targets = []
        all_sources = []

        with torch.no_grad():
            for batch in dataloader:
                x, y, source_path = batch
                y_hat = self.model(x)
                all_inputs.extend(x.cpu().numpy())
                all_predictions.extend(y_hat.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_sources.extend(source_path)

                if len(all_predictions) >= num_samples:
                    break

        print(type(all_sources))
        print(all_sources)
        print(all_sources[:5])

        all_inputs = np.concatenate(all_inputs, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        # all_sources = np.concatenate(all_sources, axis=0)

        print(type(all_inputs))
        print(type(all_predictions))
        print(type(all_targets))
        print(type(all_sources))

        print(all_inputs.shape)
        print(all_predictions.shape)
        print(all_targets.shape)
        # print(all_sources.shape)

        print(all_inputs[:5])
        print(all_predictions[:5])
        print(all_targets[:5])
        print(all_sources[:5])

        test_inputs_df = pd.DataFrame(all_inputs, columns=output_column_names)
        print(test_inputs_df.info())
        test_predictions_df = pd.DataFrame(all_predictions, columns=output_column_names)
        print(test_predictions_df.info())
        test_targets_df = pd.DataFrame(all_targets, columns=output_column_names)
        print(test_targets_df.info())

        test_sources_df = pd.DataFrame(all_sources, columns=["Source Path"])
        print(test_sources_df.info())

        # print(test_df.info())

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        save_dir = Path(save_dir, folder)
        save_dir.mkdir(exist_ok=True, parents=True)

        file_path = (
            save_dir / f"test_results.inputs.{self.model_type}_{timestamp}.parquet"
        )
        test_inputs_df.to_parquet(file_path)
        print(f"Test inputs saved to: {file_path}")

        file_path = (
            save_dir / f"test_results.predictions.{self.model_type}_{timestamp}.parquet"
        )
        test_predictions_df.to_parquet(file_path)
        print(f"Test predictions saved to: {file_path}")

        file_path = (
            save_dir / f"test_results.targets.{self.model_type}_{timestamp}.parquet"
        )
        test_targets_df.to_parquet(file_path)
        print(f"Test targets saved to: {file_path}")

        file_path = (
            save_dir / f"test_results.sources.{self.model_type}_{timestamp}.parquet"
        )
        test_sources_df.to_parquet(file_path)
        print(f"Test sources saved to: {file_path}")

        # Create visualization
        # fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5 * num_samples))
        # if num_samples == 1:
        #     axes = [axes]

        # for i in range(num_samples):
        #     input_seq = all_inputs[i]
        #     pred_seq = all_predictions[i]
        #     target_seq = all_targets[i]

        #     # Plot input sequence
        #     t_input = np.arange(self.window_size)
        #     axes[i].plot(
        #         t_input, input_seq[:, 0], "g-", label="Input Sequence", alpha=0.7
        #     )

        #     # Plot prediction and target
        #     t_pred = np.arange(self.window_size, 2 * self.window_size)
        #     axes[i].plot(
        #         t_pred, target_seq[:, 0], "b-", label="Actual Next Sequence", alpha=0.7
        #     )
        #     axes[i].plot(
        #         t_pred,
        #         pred_seq[:, 0],
        #         "r--",
        #         label="Predicted Next Sequence",
        #         alpha=0.7,
        #     )

        #     # Add vertical line to separate input and prediction
        #     axes[i].axvline(x=self.window_size, color="k", linestyle=":", alpha=0.5)

        #     axes[i].set_title(f"Sample {i+1}")
        #     axes[i].legend()
        #     axes[i].grid(True)
        #     axes[i].set_xlabel("Time Steps")
        #     axes[i].set_ylabel("Displacement")

        # plt.tight_layout()

        # save_dir = Path("./test_results_img")
        # save_dir.mkdir(exist_ok=True)

        # # plt.savefig(save_path)
        # plt.savefig(save_dir / f"test_results_{self.model_type}_{timestamp}.png")
        # plt.show()

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot comprehensive training history with multiple metrics"""
        history = self.model.history
        epochs = range(1, len(history["train_mse"]) + 1)

        # Create a figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot MSE Loss
        ax1.plot(epochs, history["train_mse"], "b-", label="Training MSE")
        ax1.plot(epochs, history["val_mse"], "r-", label="Validation MSE")
        ax1.set_title("MSE Loss Over Time")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE Loss")
        ax1.grid(True)
        ax1.legend()

        # Plot MAE
        ax2.plot(epochs, history["train_mae"], "b-", label="Training MAE")
        ax2.plot(epochs, history["val_mae"], "r-", label="Validation MAE")
        ax2.set_title("MAE Over Time")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE")
        ax2.grid(True)
        ax2.legend()

        # Plot R² Score
        ax3.plot(epochs, history["train_r2"], "b-", label="Training R²")
        ax3.plot(epochs, history["val_r2"], "r-", label="Validation R²")
        ax3.set_title("R² Score Over Time")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("R²")
        ax3.grid(True)
        ax3.legend()

        # Plot Learning Rate if available
        if len(history["learning_rates"]) > 0:
            ax4.plot(epochs, history["learning_rates"], "g-", label="Learning Rate")
            ax4.set_title("Learning Rate Over Time")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Learning Rate")
            ax4.grid(True)
            ax4.legend()
        else:
            # Plot MSE on log scale if LR not available
            ax4.semilogy(epochs, history["train_mse"], "b-", label="Training MSE (log)")
            ax4.semilogy(epochs, history["val_mse"], "r-", label="Validation MSE (log)")
            ax4.set_title("MSE Loss Over Time (Log Scale)")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Log MSE Loss")
            ax4.grid(True)
            ax4.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

        # Print final metrics
        print("\nFinal Training Metrics:")
        print(f"MSE: {history['train_mse'][-1]:.6f}")
        print(f"MAE: {history['train_mae'][-1]:.6f}")
        print(f"R²: {history['train_r2'][-1]:.6f}")

        print("\nFinal Validation Metrics:")
        print(f"MSE: {history['val_mse'][-1]:.6f}")
        print(f"MAE: {history['val_mae'][-1]:.6f}")
        print(f"R²: {history['val_r2'][-1]:.6f}")


#  End Trainer Class ----------------------------------------------------}}}
#  Main -----------------------------------------------------------------{{{


# Example usage:
def main(station_number, epochs, model, num_layers):
    # Configuration
    INPUT_COLUMNS = [
        "north_displacement_meters",
        "east_displacement_meters",
        "vert_displacement_meters",
    ]
    # MODEL_NAME = "lstm"
    # MODEL_NAME = "transformer"
    MODEL_NAME = args.model
    STATION_NUMBER = station_number
    OUTPUT_COLUMNS = INPUT_COLUMNS
    HIDDEN_DIM = 128
    NUM_LAYERS = args.num_layers
    DROPOUT = 0.2
    # EPOCHS = 25
    EPOCHS = args.epochs
    WINDOW_SIZE = 128
    # BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.2
    MODEL_LABEL = f"model_{MODEL_NAME}.station_number_{STATION_NUMBER}.window_{WINDOW_SIZE}.hidden_dim_{HIDDEN_DIM}.NUM_LAYERS_{NUM_LAYERS}.EPOCHS_{EPOCHS}"
    print(MODEL_LABEL)

    # Initialize data manager
    data_manager = WaveDataManager(
        # spec_file='wave_data_spec.parquet',
        spec_file=f"./model_input_spec_{STATION_NUMBER}.parquet",
        input_columns=INPUT_COLUMNS,
        output_columns=OUTPUT_COLUMNS,
        window_size=WINDOW_SIZE,
        batch_size=32,
    )

    # Create predictor
    lstm_models = ["lstm", "conv_lstm", "enhanced_lstm"]

    if MODEL_NAME in lstm_models:
        predictor = WavePredictor(
            model_type=MODEL_NAME,
            input_dim=len(INPUT_COLUMNS),
            model_params={
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            },
            max_epochs=EPOCHS,
        )
    elif MODEL_NAME == "transformer" or MODEL_NAME == "enhanced_transformer":
        predictor = WavePredictor(
            model_type=MODEL_NAME,
            input_dim=len(INPUT_COLUMNS),
            model_params={
                # "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            },
            max_epochs=EPOCHS,
        )

    # Prepare data
    train_loader, val_loader, test_loader = data_manager.prepare_data_splits(
        test_split=VALIDATION_SPLIT
    )

    # Train model
    model_path = predictor.train(
        train_loader,
        val_loader,
        MODEL_LABEL,
        save_dir="models",
        save_history=True,
        history_dir="training_history",
    )

    # Evaluate model
    test_results = predictor.evaluate(test_loader, model_path)
    print("Test Results:", test_results)

    # Visualize results
    predictor.save_predictions(test_loader, OUTPUT_COLUMNS, MODEL_LABEL, num_samples=3)
    # predictor.plot_training_history(save_path="training_history.png")


if __name__ == "__main__":
    # Command-line argument parser setup
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("--epochs", type=int, default=25, help="Number of Epochs")
    parser.add_argument(
        "--station_number", type=str, default="225", help="Station Number"
    )
    parser.add_argument(
        "--model", type=str, default="lstm", help="Model type, lstm or transformer"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of Model Layers"
    )
    # parser.add_argument(
    #     "--learning_optimization",
    #     type=str,
    #     default="default",
    #     help="Type of Learning Strategy",
    # )
    # parser.add_argument("--clean", type=str, default="original", help="Cleaning Level")
    # parser.add_argument("--stop_early", action=argparse.BooleanOptionalAction)

    # parser.add_argument(
    #     "--embedding_dim", type=int, default=128, help="Embedding Dimension"
    # )
    # parser.add_argument("--embedding_layers", type=int, default=1, help="Embedding Layers")
    # parser.add_argument("--comparison_type", type=str, default=None, help="Comparison Type")
    # parser.add_argument(
    #     "--tokenizer",
    #     type=str,
    #     default="bert-base-cased",
    #     help="Tokenizer <https://github.com/AkariAsai/pytorch-pretrained-BERT/blob/master/README.md>",
    # )
    args = parser.parse_args()

    main(args.station_number, args.epochs, args.model, args.num_layers)

#  End Main -------------------------------------------------------------}}}
