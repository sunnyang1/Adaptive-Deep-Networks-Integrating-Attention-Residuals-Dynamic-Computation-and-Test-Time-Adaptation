#!/usr/bin/env python3
"""
Training Script for Adaptive Deep Networks

Supports Small (1.5B), Medium (7B), and Large (50B) model sizes.
Based on paper Section 5.1 and Appendix A.2.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from models.configs import get_config, get_model_size_params
from models.adaptive_transformer import create_adaptive_transformer
from models.tokenizer import create_tokenizer


class DummyDataset(Dataset):
    """Dummy dataset for testing/training without real data."""
    
    def __init__(self, num_samples: int = 10000, seq_len: int = 2048, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]


def create_dataloader(batch_size: int = 4, seq_len: int = 2048, vocab_size: int = 32000):
    """Create training dataloader."""
    dataset = DummyDataset(seq_len=seq_len + 1, vocab_size=vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def compute_loss(model, input_ids, targets, device):
    """Compute cross-entropy loss."""
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
    return loss


def train_epoch(model, dataloader, optimizer, device, epoch, log_interval=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (input_ids, targets) in enumerate(pbar):
        loss = compute_loss(model, input_ids, targets, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train Adaptive Deep Networks')
    parser.add_argument('--model-size', type=str, default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save-every', type=int, default=1)
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("Adaptive Deep Networks - Training")
    print("=" * 60)
    print(f"Model size: {args.model_size}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Load config
    config = get_config(args.model_size)
    param_count = get_model_size_params(config)
    print(f"\nModel parameters: {param_count / 1e9:.2f}B")
    print(f"Layers: {config.num_layers}, Hidden: {config.hidden_dim}")
    
    # Create model
    print("\nInitializing model...")
    model = create_adaptive_transformer(args.model_size)
    model = model.to(device)
    print(f"Model initialized on {device}")
    
    # Create dataloader
    dataloader = create_dataloader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.output_dir, 
                f"{args.model_size}_epoch{epoch}.pt"
            )
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Final checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
