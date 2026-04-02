#!/usr/bin/env python3
"""
Base Trainer for Adaptive Deep Networks.

Provides common training functionality for all model sizes.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.configs import ModelConfig, AttnResSmallConfig, AttnResMediumConfig, AttnResLargeConfig
from src.models.adaptive_transformer import AdaptiveTransformer
from scripts.common.training import CheckpointManager, compute_loss, train_step, get_scheduler
from scripts.common.distributed import setup_distributed, cleanup_distributed, is_main_process
from scripts.common.data import HuggingFaceDataset, get_dataloader
from src.models.tokenizer import create_tokenizer, get_tokenizer_for_model


class BaseTrainer(ABC):
    """Base trainer class for all model sizes."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._get_model_config()
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.checkpoint_manager = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
        
    @abstractmethod
    def _get_model_config(self) -> ModelConfig:
        """Return model configuration for this trainer."""
        pass
    
    @abstractmethod
    def get_model_size_name(self) -> str:
        """Return model size name (small/medium/large)."""
        pass
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.args.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.args.device)
    
    def setup(self):
        """Initialize training components."""
        print(f"\n{'='*70}")
        print(f"Setting up {self.get_model_size_name().upper()} model training")
        print(f"{'='*70}")
        
        # Print config
        print(f"\nModel Configuration:")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Hidden dim: {self.config.hidden_dim}")
        print(f"  Num heads: {self.config.num_heads}")
        print(f"  Num blocks (AttnRes): {self.config.num_blocks}")
        
        # Create model
        print(f"\nBuilding model...")
        start = time.time()
        self.model = AdaptiveTransformer(self.config)
        build_time = time.time() - start
        
        params = self.model.count_parameters()
        print(f"  Parameters: {params/1e6:.1f}M")
        print(f"  Build time: {build_time:.2f}s")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup checkpoint manager
        checkpoint_dir = Path(self.args.output_dir) / 'checkpoints'
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=self.args.max_checkpoints
        )
        
        # Setup data loaders
        self._setup_data()
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.args.epochs // self.args.grad_accum
        self.scheduler = get_scheduler(
            self.optimizer,
            self.args.warmup_steps,
            total_steps
        )
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Learning rate: {self.args.lr}")
        print(f"  Warmup steps: {self.args.warmup_steps}")
        print(f"  Total scheduled steps: {total_steps}")
        print(f"  Gradient accumulation: {self.args.grad_accum}")
        print(f"  Device: {self.device}")
        print(f"  Output dir: {self.args.output_dir}")
    
    def _setup_data(self):
        """Setup training and validation data loaders using HuggingFace datasets."""
        # Tokenizer configuration
        tokenizer_name = self.args.tokenizer_name or get_tokenizer_for_model(self.get_model_size_name())
        hf_token = self.args.hf_token  # For gated models like Llama-2
        
        # Create tokenizer
        tokenizer = create_tokenizer(
            tokenizer_name=tokenizer_name,
            vocab_size=self.config.vocab_size,
            token=hf_token,
        )
        
        # Dataset configuration from args
        dataset_name = self.args.dataset_name
        dataset_config = self.args.dataset_config
        streaming = self.args.streaming
        max_train_samples = self.args.max_train_samples or self.args.train_samples
        max_val_samples = self.args.max_val_samples or self.args.val_samples
        
        print(f"\nDataset Configuration:")
        print(f"  Dataset: {dataset_name}")
        if dataset_config:
            print(f"  Config: {dataset_config}")
        print(f"  Tokenizer: {tokenizer_name}")
        print(f"  Streaming: {streaming}")
        print(f"  Max train samples: {max_train_samples}")
        print(f"  Max val samples: {max_val_samples}")
        
        # Create datasets
        train_dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split="train",
            seq_len=self.args.seq_len,
            tokenizer=tokenizer,
            max_samples=max_train_samples,
            streaming=streaming,
        )
        
        val_dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split="validation" if not streaming else "train",  # Some datasets only have train
            seq_len=self.args.seq_len,
            tokenizer=tokenizer,
            max_samples=max_val_samples,
            streaming=streaming,
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=not streaming,  # Cannot shuffle streaming datasets easily
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def train_epoch(self) -> float:
        """Train one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated_steps = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            loss = train_step(
                model=self.model,
                batch=batch,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                gradient_accumulation_steps=self.args.grad_accum
            )
            
            accumulated_steps += 1
            total_loss += loss
            
            # Perform optimizer step only after accumulating enough gradients
            if accumulated_steps >= self.args.grad_accum:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                accumulated_steps = 0
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Log periodically
                if self.global_step % self.args.log_interval == 0:
                    self._log_step(loss)
        
        # Handle any remaining accumulated gradients
        if accumulated_steps > 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                loss = compute_loss(
                    model=self.model,
                    batch=batch,
                    criterion=self.criterion,
                    device=self.device
                )
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _log_step(self, loss: float):
        """Log training step."""
        if is_main_process():
            print(f"Step {self.global_step}: loss = {loss:.4f}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        metrics = {
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'global_step': self.global_step,
        }
        
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            loss=self.history['train_loss'][-1] if self.history['train_loss'] else 0.0,
            metrics=metrics,
            config=self.config.__dict__,
            is_best=is_best
        )
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            # Check if best
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            print(f"\nEpoch {self.current_epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Best: {'✅' if is_best else '❌'}")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best val loss: {self.best_loss:.4f}")
        print(f"{'='*70}")
        
        # Save final results
        self._save_results()
    
    def _save_results(self):
        """Save training results."""
        results = {
            'model_size': self.get_model_size_name(),
            'config': self.config.__dict__,
            'training_args': vars(self.args),
            'history': self.history,
            'best_loss': self.best_loss,
            'total_steps': self.global_step,
        }
        
        output_file = Path(self.args.output_dir) / 'training_results.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def get_common_parser() -> argparse.ArgumentParser:
    """Get argument parser with common arguments."""
    parser = argparse.ArgumentParser(description='Train Adaptive Deep Networks')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--max-checkpoints', type=int, default=3, help='Max checkpoints to keep')
    
    # Data configuration
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--train-samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--val-samples', type=int, default=1000, help='Validation samples')
    
    # Paths
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Execution
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    
    # Dataset configuration
    parser.add_argument('--dataset-name', type=str, default='openwebtext',
                       help='HuggingFace dataset name (e.g., openwebtext, HuggingFaceFW/fineweb-edu)')
    parser.add_argument('--dataset-config', type=str, default=None,
                       help='Dataset configuration name')
    parser.add_argument('--streaming', action='store_true', default=True,
                       help='Use streaming mode for datasets (no local download)')
    parser.add_argument('--max-train-samples', type=int, default=None,
                       help='Maximum training samples (defaults to train-samples)')
    parser.add_argument('--max-val-samples', type=int, default=None,
                       help='Maximum validation samples (defaults to val-samples)')
    
    # Tokenizer configuration
    parser.add_argument('--tokenizer-name', type=str, default=None,
                       help='Tokenizer name (e.g., gpt2, meta-llama/Llama-2-7b-hf, meta-llama/Meta-Llama-3-8B)')
    parser.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace API token (for gated models like Llama-2)')
    
    return parser
