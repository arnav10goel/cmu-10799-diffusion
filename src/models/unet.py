"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # TODO: build your own unet architecture here
        # Pro tips: remember to take care of the time embeddings!
        # 1. Time embedding
        self.time_embed_dim = base_channels * 4
        self.time_mlp = TimestepEmbedding(self.time_embed_dim) # TODO: adjust input dim if necessary (right now it goes 512 -> 2048 -> 512)
        
        # 2. Initial convolution
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1) # (batch_size, base_channels, 64, 64)
        
        # 3. Downsampling path (Encoder)
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        input_block_chans = [ch]  # Keep track of channels for skip connections
        ds = 1  # current downsampling factor (assuming input is 64x64 or similar)
        
        # Tracking resolution (assuming input resolution is 64 for determining attention)
        curr_res = 64
        for level, mult in enumerate(channel_mult):
            level_out_channels = base_channels * mult # (eg: [1,2,2,4] -> [128,256,256,512] if base_channels=128)
            
            # In each level, add num_res_blocks ResBlocks
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        in_channels=ch,
                        out_channels=level_out_channels,
                        time_embed_dim=self.time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = level_out_channels
                
                # Add attention if at the right resolution
                if curr_res in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                        )
                    )
                
                self.down_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
                
            # Add Downsample (except for the last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
                input_block_chans.append(ch)
                ds *= 2
                curr_res //= 2
                
        # 4. Middle Block (Bottleneck)
        self.middle_block = nn.Sequential(
            ResBlock(
                in_channels=ch,
                out_channels=ch,
                time_embed_dim=self.time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                channels=ch,
                num_heads=num_heads,
            ),
            ResBlock(
                in_channels=ch,
                out_channels=ch,
                time_embed_dim=self.time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        
        # 5. Upsampling Path (Decoder)
        self.up_blocks = nn.ModuleList()
        
        # Reverse channel_mult to go up
        for level, mult in enumerate(reversed(channel_mult)):
            level_out_channels = base_channels * mult
            
            # We add num_res_blocks + 1 blocks in the decoder to process skips
            # Since we had num_res + 1 blocks in the encoder at each level (including downsample)
            for i in range(num_res_blocks + 1):
                skip_channels = input_block_chans.pop()
                layers = [
                    ResBlock(
                        in_channels=ch + skip_channels,
                        out_channels=level_out_channels,
                        time_embed_dim=self.time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = level_out_channels
                
                # Add attention if at the right resolution
                if curr_res in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                        )
                    )
                    
                self.up_blocks.append(nn.Sequential(*layers))
                
            # Add Upsample (except for the last level)
            if level != len(channel_mult) - 1:
                self.up_blocks.append(Upsample(ch))
                ds //= 2
                curr_res *= 2
                
        # 6. Final normalization and convolution
        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement the forward pass of the unet
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width) (B, C, H, W)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,) (B,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width) (B, C, H, W)
        """

        # 1. Time embedding
        t_emb = self.time_mlp(t)  # (B, time_embed_dim)
        
        # 2. Initial convolution
        h = self.head(x)  # (B, base_channels, H, W)
        heads = [h]  # For skip connections
        
        # 3. Downsampling path (Encoder)
        for module in self.down_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            heads.append(h)
            
        # 4. Middle Block (Bottleneck)
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # 5. Upsampling Path (Decoder)
        for module in self.up_blocks:
            if isinstance(module, nn.Sequential):
                # Pop the skip connection to concatenate
                skip = heads.pop()
                h = torch.cat([h, skip], dim=1)
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
                
        # 6. Final normalization and convolution
        out = self.out(h)
        
        return out

# =============================================================================
def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
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


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
