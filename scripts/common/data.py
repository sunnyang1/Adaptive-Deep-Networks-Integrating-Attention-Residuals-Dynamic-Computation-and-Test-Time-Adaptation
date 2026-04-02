"""
Data Loading Utilities

Shared dataset and dataloader implementations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Iterator


class DummyDataset(Dataset):
    """
    Dummy dataset for testing.
    
    Generates random data of specified shape.
    """
    
    def __init__(
        self,
        size: int = 1000,
        seq_len: int = 512,
        vocab_size: int = 32000,
        seed: int = 42
    ):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> dict:
        # Deterministic per-sample generator for reproducibility
        g = torch.Generator().manual_seed(self.seed + idx)
        
        # Generate random sequence
        input_ids = torch.randint(
            0, self.vocab_size,
            (self.seq_len,),
            generator=g
        )
        
        # Shift for labels (next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0  # End token
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StreamingTextDataset(Dataset):
    """
    Streaming dataset for large text corpora.
    
    Loads data on-the-fly to avoid memory issues.
    """
    
    def __init__(
        self,
        data_iterator: Iterator,
        seq_len: int = 2048,
        max_samples: Optional[int] = None
    ):
        self.data_iterator = data_iterator
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.buffer = []
        self._fill_buffer()
    
    def _fill_buffer(self):
        """Fill buffer from iterator."""
        try:
            while len(self.buffer) < (self.max_samples or 1000):
                item = next(self.data_iterator)
                self.buffer.append(item)
        except StopIteration:
            pass
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self.buffer):
            raise IndexError(f"Index {idx} out of range")
        
        text = self.buffer[idx]
        
        # Tokenize (simplified - should use actual tokenizer)
        tokens = text.split()[:self.seq_len]
        input_ids = torch.tensor([hash(t) % 32000 for t in tokens], dtype=torch.long)
        
        # Pad to seq_len
        if len(input_ids) < self.seq_len:
            input_ids = torch.cat([
                input_ids,
                torch.zeros(self.seq_len - len(input_ids), dtype=torch.long)
            ])
        
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create dataloader with standard settings.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA
    
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
