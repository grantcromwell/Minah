import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import mlflow
import mlflow.pytorch
import numpy as np
from datetime import datetime
from pathlib import Path

from .models.transformer_model import TransformerModel
from .data.dataset import TradingDataset
from .utils.metrics import calculate_sharpe_ratio, max_drawdown

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class TradingLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
    
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        policy_l = self.policy_loss(policy_pred, policy_target)
        value_l = self.value_loss(value_pred, value_target)
        return self.alpha * policy_l + (1 - self.alpha) * value_l

def train_epoch(rank, model, train_loader, criterion, optimizer, scaler, epoch, log_interval=10):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, policy_target, value_target) in enumerate(train_loader):
        data, policy_target, value_target = data.to(rank), policy_target.to(rank), value_target.to(rank)
        
        optimizer.zero_grad()
        
        with autocast():
            policy_pred, value_pred = model(data)
            loss = criterion(policy_pred, value_pred, policy_target, value_target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0 and batch_idx > 0 and rank == 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                  f'ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model, eval_loader, criterion, rank):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, policy_target, value_target in eval_loader:
            data, policy_target, value_target = data.to(rank), policy_target.to(rank), value_target.to(rank)
            
            with autocast():
                policy_pred, value_pred = model(data)
                loss = criterion(policy_pred, value_pred, policy_target, value_target)
            
            total_loss += loss.item()
            all_predictions.append((policy_pred.argmax(dim=1), value_pred))
            all_targets.append((policy_target, value_target))
    
    return total_loss / len(eval_loader), all_predictions, all_targets

def train(rank, world_size, config):
    setup(rank, world_size)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    
    # Initialize model and move to GPU
    model = TransformerModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        output_dim=config['output_dim']
    ).to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Initialize optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3, verbose=True
    )
    
    # Loss function
    criterion = TradingLoss(alpha=0.7)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Load datasets
    train_dataset = TradingDataset(config['train_data_path'], seq_len=config['seq_len'])
    val_dataset = TradingDataset(config['val_data_path'], seq_len=config['seq_len'])
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize MLflow
    if rank == 0:
        mlflow.set_tracking_uri(config['mlflow_uri'])
        mlflow.set_experiment(config['experiment_name'])
        mlflow.start_run()
        mlflow.log_params(config)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config['epochs'] + 1):
        train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_epoch(rank, model, train_loader, criterion, optimizer, scaler, epoch)
        
        # Evaluate on validation set
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, rank)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        if rank == 0:
            # Calculate additional metrics
            policy_preds = torch.cat([p[0] for p in val_preds])
            value_preds = torch.cat([p[1] for p in val_preds])
            policy_targets = torch.cat([t[0] for t in val_targets])
            value_targets = torch.cat([t[1] for t in val_targets])
            
            accuracy = (policy_preds == policy_targets).float().mean().item()
            sharpe_ratio = calculate_sharpe_ratio(value_preds.cpu().numpy())
            mdd = max_drawdown(value_preds.cpu().numpy())
            
            # Log to MLflow
            mlflow.log_metrics({
                'val_loss': val_loss,
                'val_accuracy': accuracy,
                'val_sharpe': sharpe_ratio,
                'val_mdd': mdd,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, f"{config['model_dir']}/best_model.pt")
                
                # Log model to MLflow
                mlflow.pytorch.log_model(model.module, "model")
    
    if rank == 0:
        mlflow.end_run()
    
    cleanup()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train trading model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Set up distributed training
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
