"""
Refactored Training Script

Demonstrates the improved architecture using scripts/common/ modules.
Replaces train_model.py, train_h20.py, train_streaming.py

Usage:
    python scripts/training/train_refactored.py --model-size small
    python scripts/training/train_refactored.py --model-size medium --distributed
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm

# Import shared modules
from scripts.common.paths import get_default_paths, ensure_directories
from scripts.common.distributed import setup_distributed, cleanup_distributed, is_main_process
from scripts.common.training import CheckpointManager, compute_loss, train_step
from scripts.common.data import DummyDataset, get_dataloader
from experiments.common import Environment
from experiments.common import ExperimentConfig, get_device, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Adaptive Deep Networks (Refactored)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU training
  %(prog)s --model-size small --epochs 3
  
  # Multi-GPU distributed training
  %(prog)s --model-size medium --distributed
  
  # Quick test run
  %(prog)s --model-size small --quick
        """
    )
    
    # Model configuration
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Model size (small: 2.2B, medium: 8.7B, large: 27B)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Max training steps (overrides epochs)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size per GPU (auto if not specified)')
    parser.add_argument('--seq-len', type=int, default=1024,
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--grad-accum', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (auto-detect if not specified)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    # Execution
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic mode')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval (steps)')
    parser.add_argument('--save-interval', type=int, default=500,
                       help='Checkpoint save interval (steps)')
    parser.add_argument('--eval-interval', type=int, default=1000,
                       help='Evaluation interval (steps)')
    
    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (reduced steps)')
    
    return parser.parse_args()


def get_model_config(size: str):
    """Get model configuration for size."""
    configs = {
        'small': {
            'vocab_size': 32000,
            'hidden_dim': 2048,
            'num_layers': 32,
            'num_heads': 32,
            'batch_size': 8,
            'description': '2.2B parameters'
        },
        'medium': {
            'vocab_size': 32000,
            'hidden_dim': 4096,
            'num_layers': 32,
            'num_heads': 32,
            'batch_size': 2,
            'description': '8.7B parameters'
        },
        'large': {
            'vocab_size': 32000,
            'hidden_dim': 5120,
            'num_layers': 64,
            'num_heads': 40,
            'batch_size': 1,
            'description': '27B parameters'
        },
    }
    return configs[size]


def create_model(config: dict, device: torch.device):
    """Create model from config."""
    # For demonstration, use simple transformer
    # In production, import from src.models
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_dim'] * 4,
            batch_first=True
        ),
        num_layers=config['num_layers']
    )
    return model.to(device)


def main():
    args = parse_args()
    
    # Detect environment
    env_name = Environment.get_name()
    
    # Setup paths
    default_paths = get_default_paths()
    output_dir = Path(args.output_dir) if args.output_dir else default_paths['checkpoints']
    ensure_directories({'output': output_dir})
    
    # Setup distributed
    rank = 0
    world_size = 1
    local_rank = 0
    
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
    
    # Setup device and logging
    device = get_device(args.device)
    logger = get_logger('train')
    
    if is_main_process(rank):
        logger.info("="*60)
        logger.info("Adaptive Deep Networks - Training")
        logger.info("="*60)
        logger.info(f"Environment: {env_name}")
        logger.info(f"Model size: {args.model_size}")
        logger.info(f"Device: {device}")
        logger.info(f"Distributed: {args.distributed} (world_size={world_size})")
        logger.info(f"Output: {output_dir}")
    
    # Get model config
    model_config = get_model_config(args.model_size)
    batch_size = args.batch_size or model_config['batch_size']
    
    # Quick mode adjustments
    if args.quick:
        args.epochs = 1
        args.max_steps = 100
        logger.info("Quick mode: Reduced epochs and steps")
    
    if is_main_process(rank):
        logger.info(f"Model: {model_config['description']}")
        logger.info(f"Batch size per GPU: {batch_size}")
        logger.info(f"Global batch size: {batch_size * world_size * args.grad_accum}")
        logger.info(f"Sequence length: {args.seq_len}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info("")
    
    # Create model
    if is_main_process(rank):
        logger.info("Creating model...")
    
    model = create_model(model_config, device)
    
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Create dataset and dataloader
    dataset = DummyDataset(
        size=10000 if not args.quick else 100,
        seq_len=args.seq_len,
        vocab_size=model_config['vocab_size'],
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    
    if args.resume:
        if is_main_process(rank):
            logger.info(f"Resuming from {args.resume}")
        checkpoint = checkpoint_manager.load(
            model, optimizer, Path(args.resume)
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('metrics', {}).get('global_step', 0)
    
    # Training loop
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(start_epoch, args.epochs):
        if is_main_process(rank):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            logger.info("-"*40)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            dataloader,
            disable=not is_main_process(rank),
            desc=f"Epoch {epoch+1}"
        )
        
        for batch in progress:
            loss = train_step(
                model, batch, optimizer, criterion, device
            )
            
            # Gradient accumulation
            if (num_batches + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress.set_postfix({'loss': f'{loss:.4f}', 'step': global_step})
            
            # Logging
            if global_step % args.log_interval == 0 and is_main_process(rank):
                logger.info(
                    f"Step {global_step} | Loss: {loss:.4f}",
                    extra={'extras': {'step': global_step, 'loss': loss}}
                )
            
            # Save checkpoint
            if global_step % args.save_interval == 0 and is_main_process(rank):
                checkpoint_manager.save(
                    model,
                    optimizer,
                    epoch,
                    loss,
                    metrics={'global_step': global_step, 'epoch': epoch}
                )
            
            # Check max steps
            if args.max_steps and global_step >= args.max_steps:
                if is_main_process(rank):
                    logger.info(f"Reached max steps ({args.max_steps})")
                break
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        if is_main_process(rank):
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            checkpoint_manager.save(
                model,
                optimizer,
                epoch,
                avg_loss,
                metrics={'global_step': global_step, 'epoch': epoch},
                is_best=(epoch == start_epoch)  # Simplified
            )
        
        # Check max steps
        if args.max_steps and global_step >= args.max_steps:
            break
    
    # Cleanup
    if args.distributed:
        cleanup_distributed()
    
    if is_main_process(rank):
        logger.info("\n" + "="*60)
        logger.info("Training complete!")
        logger.info(f"Final step: {global_step}")
        logger.info(f"Checkpoints: {output_dir / 'checkpoints'}")
        logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
