"""
Models module for cmu-10799-diffusion.

This module contains the neural network architectures used for
diffusion models and flow matching.
"""

from .unet import UNet
from .dit import DiT
from .blocks import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


def create_model_from_config(config: dict):
    """
    Factory to create a model from config.

    Backward-compatible with existing UNet-only configs.
    Use `model.name: unet|dit` (or `model.type`) to select architecture.
    """
    model_config = config['model']
    data_config = config['data']

    model_name = str(model_config.get('name', model_config.get('type', 'unet'))).lower()

    if model_name == 'unet':
        return UNet(
            in_channels=data_config['channels'],
            out_channels=data_config['channels'],
            base_channels=model_config['base_channels'],
            channel_mult=tuple(model_config['channel_mult']),
            num_res_blocks=model_config['num_res_blocks'],
            attention_resolutions=model_config['attention_resolutions'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            use_scale_shift_norm=model_config['use_scale_shift_norm'],
        )

    if model_name == 'dit':
        return DiT(
            input_size=data_config['image_size'],
            patch_size=model_config.get('patch_size', 4),
            in_channels=data_config['channels'],
            hidden_size=model_config.get('hidden_size', 384),
            depth=model_config.get('depth', 6),
            num_heads=model_config.get('num_heads', 6),
            mlp_ratio=model_config.get('mlp_ratio', 4.0),
            class_dropout_prob=model_config.get('class_dropout_prob', 0.1),
            learn_sigma=model_config.get('learn_sigma', False),
        )

    raise ValueError(f"Unknown model.name/model.type '{model_name}'. Supported: 'unet', 'dit'.")

__all__ = [
    # Main model
    'UNet',
    'DiT',
    'create_model_from_config',
    # Building blocks
    'SinusoidalPositionalEmbedding',
    'TimestepEmbedding', 
    'ResBlock',
    'AttentionBlock',
    'Downsample',
    'Upsample',
    'GroupNorm32',
]
