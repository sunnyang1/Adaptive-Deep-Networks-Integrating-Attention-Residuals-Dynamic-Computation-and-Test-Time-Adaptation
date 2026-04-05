#!/usr/bin/env python3
"""
Unified Training Script

Demonstrates the improved architecture using scripts/common/ modules.

Replaces:
- train_model.py
- train_h20.py
- train_streaming.py

Usage:
    python scripts/training/train_unified.py --model-size small
    python scripts/training/train_unified.py --model-size medium --distributed
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

# Import shared modules
from scripts.common.paths import add_project_to_path, get_default_paths, ensure_directories
from scripts.common.distributed import setup_distributed, cleanup_distributed, is_main_process
from scripts.common.training import CheckpointManager, compute_loss, train_step
from scripts.common.data import DummyDataset, get_dataloader

# Import model
from src.models.configs import get_config
from src.models.adaptive_transformer import AdaptiveTransformer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Adaptive Deep Networks')
    
    # Model configuration
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Model size')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto if not specified)')
    parser.add_argument('--seq-len', type=int, default=1024,
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Warmup steps')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (auto-detect if not specified)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory')
    
    # Execution
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Checkpoint save interval')
    
    return parser.parse_args()


def get_model_config_with_batch(size: str):
    """Get model configuration for size with batch size."""
    config = get_config(size)
    batch_sizes = {
        'small': 8,
        'medium': 2,
        'large': 1,
    }
    return config, batch_sizes[size]


def main():
    args = parse_args()
    
    # Setup paths
    default_paths = get_default_paths()
    output_dir = Path(args.output_dir) if args.output_dir else default_paths['checkpoints']
    ensure_directories({'output': output_dir})
    
    # Setup distributed
    rank = 0
    world_size = 1
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        if is_main_process(rank):
            print(f"Distributed training: rank={rank}, world_size={world_size}")
    
    # Setup device
    from experiments.common import get_device
    device = get_device(args.device)
    
    if is_main_process(rank):
        print("="*60)
        print("Unified Training Script")
        print("="*60)
        print(f"Model size: {args.model_size}")
        print(f"Device: {device}")
        print(f"Output: {output_dir}")
        print(f"Distributed: {args.distributed}")
        print()
    
    # Get model config
    config, default_batch = get_model_config_with_batch(args.model_size)
    batch_size = args.batch_size or default_batch
    
    if is_main_process(rank):
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {args.seq_len}")
        print()
    
    # Create model
    if is_main_process(rank):
        print("Creating model...")
    
    model = AdaptiveTransformer(config)
    
    model = model.to(device)
    
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1
    )
    
    # Create dataset and dataloader
    dataset = DummyDataset(
        size=1000,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
        seed=args.seed
    )
    
    dataloader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=output_dir / 'checkpoints',
        max_checkpoints=5
    )
    
    # Training loop
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    
    for epoch in range(args.epochs):
        if is_main_process(rank):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-"*40)
        
        model.train()
        epoch_loss = 0.0
        
        progress = tqdm(dataloader, disable=not is_main_process(rank))
        for batch in progress:
            loss = train_step(model, batch, optimizer, criterion, device)
            
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss
            global_step += 1
            
            # Update progress bar
            progress.set_postfix({'loss': f'{loss:.4f}'})
            
            # Save checkpoint
            if global_step % args.save_interval == 0 and is_main_process(rank):
                checkpoint_manager.save(
                    model,
                    optimizer,
                    epoch,
                    loss,
                    metrics={'global_step': global_step}
                )
        
        avg_loss = epoch_loss / len(dataloader)
        
        if is_main_process(rank):
            print(f"Average loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            checkpoint_manager.save(
                model,
                optimizer,
                epoch,
                avg_loss,
                metrics={'epoch': epoch},
                is_best=(epoch == 0)  # Simplified
            )
    
    # Cleanup
    if args.distributed:
        cleanup_distributed()
    
    if is_main_process(rank):
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
        print("="*60)


if __name__ == '__main__':
    main()
