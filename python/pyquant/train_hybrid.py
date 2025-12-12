import os
import time
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import mlflow
import mlflow.pytorch

from .models.hybrid_transformer import HybridTransformerModel
from .data.dataset import TradingDataset, create_data_loaders
from .utils.metrics import ModelEvaluator, compute_all_metrics
from .configs.train_config import load_config, update_config

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TradingLoss(nn.Module):
    """
    Custom loss function for trading that combines policy and value losses.
    """
    def __init__(self, alpha: float = 0.7, gamma: float = 2.0, label_smoothing: float = 0.1):
        """
        Initialize the trading loss.
        
        Args:
            alpha: Weight for policy loss (1-alpha for value loss)
            gamma: Focal loss gamma (if using focal loss)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.eps = 1e-8
    
    def forward(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the combined policy and value loss.
        
        Args:
            policy_pred: Predicted policy logits (batch_size, num_classes)
            value_pred: Predicted values (batch_size,)
            policy_target: Target policy (batch_size, num_classes)
            value_target: Target values (batch_size,)
            
        Returns:
            Combined loss value
        """
        # Policy loss with label smoothing and focal loss
        if self.label_smoothing > 0:
            # Apply label smoothing
            policy_target = (1 - self.label_smoothing) * policy_target + \
                          self.label_smoothing / policy_target.size(1)
        
        # Focal loss for policy
        log_pt = F.log_softmax(policy_pred, dim=-1)
        pt = torch.exp(log_pt)
        policy_loss = -((1 - pt) ** self.gamma) * log_pt * policy_target
        policy_loss = policy_loss.sum(dim=1).mean()
        
        # Value loss (MSE with Huber loss)
        value_loss = F.smooth_l1_loss(value_pred, value_target)
        
        # Combine losses
        total_loss = self.alpha * policy_loss + (1 - self.alpha) * value_loss
        
        return total_loss, {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

def setup(rank: int, world_size: int):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()

def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create the model based on the configuration."""
    model_config = config["model"]
    
    # Create the hybrid transformer model
    model = HybridTransformerModel(
        input_dim=model_config["input_dim"],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
        output_dim=model_config["output_dim"],
        use_gaf=model_config["use_gaf"],
        gaf_hidden_dims=model_config.get("gaf_hidden_dims", [64, 128]),
        gaf_method=model_config.get("gaf_method", "summation"),
        use_transformer=model_config["use_transformer"],
        activation=model_config.get("activation", "gelu"),
        use_pos_encoding=model_config.get("use_pos_encoding", True),
        gaf_output_ratio=model_config.get("gaf_output_ratio", 0.5)
    )
    
    return model

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    epoch: int,
    device: torch.device,
    rank: int = 0,
    log_interval: int = 10,
    use_amp: bool = True
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Unpack batch
        if len(batch) == 2:
            x, (policy_target, value_target) = batch
        else:
            x, policy_target, value_target = batch
        
        # Move data to device
        x = x.to(device, non_blocking=True)
        policy_target = policy_target.to(device, non_blocking=True)
        value_target = value_target.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            policy_pred, value_pred = model(x)
            loss, loss_dict = criterion(policy_pred, value_pred, policy_target, value_target)
        
        # Backward pass and optimize
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_policy_loss += loss_dict["policy_loss"]
        total_value_loss += loss_dict["value_loss"]
        
        # Log progress
        if batch_idx % log_interval == 0 and rank == 0:
            ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.6f} | Policy: {loss_dict["policy_loss"]:.6f} | '
                  f'Value: {loss_dict["value_loss"]:.6f} | ms/batch: {ms_per_batch:.1f}')
    
    # Calculate average metrics
    avg_loss = total_loss / len(train_loader)
    avg_policy_loss = total_policy_loss / len(train_loader)
    avg_value_loss = total_value_loss / len(train_loader)
    
    return {
        'train_loss': avg_loss,
        'train_policy_loss': avg_policy_loss,
        'train_value_loss': avg_value_loss
    }

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int = 0
) -> Dict[str, float]:
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    all_policy_preds = []
    all_policy_targets = []
    all_value_preds = []
    all_value_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch
            if len(batch) == 2:
                x, (policy_target, value_target) = batch
            else:
                x, policy_target, value_target = batch
            
            # Move data to device
            x = x.to(device, non_blocking=True)
            policy_target = policy_target.to(device, non_blocking=True)
            value_target = value_target.to(device, non_blocking=True)
            
            # Forward pass
            policy_pred, value_pred = model(x)
            loss, loss_dict = criterion(policy_pred, value_pred, policy_target, value_target)
            
            # Update metrics
            total_loss += loss.item()
            total_policy_loss += loss_dict["policy_loss"]
            total_value_loss += loss_dict["value_loss"]
            
            # Store predictions and targets
            all_policy_preds.append(torch.argmax(policy_pred, dim=1).cpu().numpy())
            all_policy_targets.append(torch.argmax(policy_target, dim=1).cpu().numpy())
            all_value_preds.append(value_pred.cpu().numpy())
            all_value_targets.append(value_target.cpu().numpy())
    
    # Calculate average metrics
    avg_loss = total_loss / len(val_loader)
    avg_policy_loss = total_policy_loss / len(val_loader)
    avg_value_loss = total_value_loss / len(val_loader)
    
    # Calculate additional metrics
    all_policy_preds = np.concatenate(all_policy_preds)
    all_policy_targets = np.concatenate(all_policy_targets)
    all_value_preds = np.concatenate(all_value_preds)
    all_value_targets = np.concatenate(all_value_targets)
    
    # Calculate accuracy
    policy_accuracy = (all_policy_preds == all_policy_targets).mean()
    
    # Calculate R-squared for value prediction
    value_var = np.var(all_value_targets)
    if value_var > 0:
        value_r2 = 1 - np.mean((all_value_preds - all_value_targets) ** 2) / value_var
    else:
        value_r2 = 0.0
    
    metrics = {
        'val_loss': avg_loss,
        'val_policy_loss': avg_policy_loss,
        'val_value_loss': avg_value_loss,
        'val_policy_accuracy': policy_accuracy,
        'val_value_r2': value_r2
    }
    
    return metrics

def train(
    rank: int,
    world_size: int,
    config: Dict[str, Any],
    use_amp: bool = True
):
    """Main training function."""
    # Setup distributed training
    if world_size > 1:
        setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    set_seed(config["training"]["seed"] + rank)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=config["data"]["data_path"],
        feature_columns=config["data"]["feature_columns"],
        target_columns=config["data"]["target_columns"],
        seq_len=config["data"]["seq_len"],
        batch_size=config["training"]["batch_size"],
        val_split=config["data"].get("val_split", 0.1),
        test_split=config["data"].get("test_split", 0.1),
        normalize=config["data"].get("normalize", True),
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=config["training"].get("pin_memory", True),
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size
    )
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Wrap with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Create loss function
    criterion = TradingLoss(
        alpha=config["training"].get("alpha", 0.7),
        gamma=config["training"].get("gamma", 2.0),
        label_smoothing=config["training"].get("label_smoothing", 0.1)
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"].get("weight_decay", 1e-5)
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["scheduler"].get("factor", 0.5),
        patience=config["scheduler"].get("patience", 5),
        verbose=True
    )
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler(enabled=use_amp)
    
    # Initialize MLflow
    if rank == 0 and config["logging"].get("use_mlflow", True):
        mlflow.set_tracking_uri(config["logging"].get("tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(config["logging"].get("experiment_name", "trading_transformer"))
        mlflow.start_run()
        
        # Log hyperparameters
        mlflow.log_params({
            "model": config["model"]["_name"],
            "d_model": config["model"]["d_model"],
            "nhead": config["model"]["nhead"],
            "num_layers": config["model"]["num_layers"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["optimizer"]["lr"],
            "weight_decay": config["optimizer"].get("weight_decay", 1e-5),
            "use_amp": use_amp,
            "use_gaf": config["model"].get("use_gaf", True),
            "use_transformer": config["model"].get("use_transformer", True)
        })
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
            device=device,
            rank=rank,
            log_interval=config["logging"].get("log_interval", 10),
            use_amp=use_amp
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            rank=rank
        )
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Log metrics
        if rank == 0 and config["logging"].get("use_mlflow", True):
            mlflow.log_metrics({
                **train_metrics,
                **val_metrics,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            }, step=epoch)
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss and rank == 0:
            best_val_loss = val_metrics['val_loss']
            early_stop_counter = 0
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'config': config
            }
            
            os.makedirs(config["training"]["model_dir"], exist_ok=True)
            torch.save(checkpoint, os.path.join(config["training"]["model_dir"], 'best_model.pt'))
            
            if config["logging"].get("use_mlflow", True):
                mlflow.pytorch.log_model(
                    pytorch_model=model.module if world_size > 1 else model,
                    artifact_path="model",
                    registered_model_name=config["logging"].get("model_name", "trading_transformer")
                )
                
                # Log best metrics
                mlflow.log_metrics({
                    'best_val_loss': best_val_loss,
                    'best_epoch': epoch
                }, step=epoch)
        else:
            early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= config["training"].get("patience", 10):
                print(f'Early stopping at epoch {epoch}')
                break
    
    # Final evaluation on test set
    if rank == 0 and test_loader is not None:
        # Load best model
        checkpoint = torch.load(os.path.join(config["training"]["model_dir"], 'best_model.pt'))
        
        if world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        test_metrics = validate(
            model=model,
            val_loader=test_loader,
            criterion=criterion,
            device=device,
            rank=rank
        )
        
        # Log test metrics
        if config["logging"].get("use_mlflow", True):
            test_metrics = {f'test_{k}': v for k, v in test_metrics.items()}
            mlflow.log_metrics(test_metrics)
            
            # Log model artifacts
            mlflow.log_artifacts(config["training"]["model_dir"])
            
            # End MLflow run
            mlflow.end_run()
    
    # Clean up distributed training
    if world_size > 1:
        cleanup()

def main():
    """Main function for training the model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Hybrid Transformer Model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision training')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Update config with command line arguments
    config["training"]["use_amp"] = args.use_amp
    
    # Set device
    if args.gpus > 0 and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        args.gpus = 0
    
    # Create model directory
    os.makedirs(config["training"]["model_dir"], exist_ok=True)
    
    # Train model
    if args.gpus <= 1:
        train(0, 1, config, use_amp=args.use_amp)
    else:
        mp.spawn(
            train,
            args=(args.gpus, config, args.use_amp),
            nprocs=args.gpus,
            join=True
        )

if __name__ == '__main__':
    main()
