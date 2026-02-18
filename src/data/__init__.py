"""
Data module for cmu-10799-diffusion.

This module contains dataset loading and preprocessing utilities.
"""
from torch.utils.data import DataLoader

from .celeba import (
    CelebADataset,
    create_dataloader_from_config as create_celeba_dataloader,
    unnormalize,
    normalize,
    make_grid,
    save_image,
)

# 2. Reflow Dataset and DataLoader
from .reflow_dataset import ReflowDataset, create_reflow_dataloader

def create_dataloader_from_config(config: dict, split: str = 'train') -> DataLoader:
    """
    Master factory function that routes to the correct dataset loader
    based on config['data']['dataset'].
    """
    data_config = config.get('data', {})
    dataset_type = data_config.get('dataset', 'celeba') # Default to celeba

    # === ROUTING LOGIC ===
    if dataset_type == 'reflow':
        print(f"Data Factory: Creating Reflow DataLoader from {data_config['root']}")
        return create_reflow_dataloader(config, split)
    
    elif dataset_type == 'celeba':
        # Delegate to the function in celeba.py
        return create_celeba_dataloader(config, split)
    
    else:
        # Fallback / Default
        print(f"Data Factory: Unknown dataset type '{dataset_type}', defaulting to CelebA logic.")
        return create_celeba_dataloader(config, split)

__all__ = [
    'CelebADataset',
    'create_dataloader',
    'create_dataloader_from_config',
    'unnormalize',
    'normalize',
    'make_grid',
    'save_image',
]
