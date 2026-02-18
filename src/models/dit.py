import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from .blocks import TimestepEmbedding

def modulate(x, shift, scale):
    """
    Modulate the normalized input x using shift and scale.
    x: (B, N, D)
    shift, scale: (B, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with Adaptive Layer Norm (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # adaLN modulation: regresses 6 parameters (shift, scale, gate) * 2 (for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: Input tensor (B, N, D)
        c: Conditioning vector (B, D) - usually the time embedding
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Attention Block
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. MLP Block
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) architecture.
    """
    def __init__(
        self,
        input_size=64,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=False,
    ):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by patch_size ({patch_size})")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.class_dropout_prob = class_dropout_prob

        # 1. Patch Embedding
        # Input: (B, C, H, W) -> (B, N, D)
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # 2. Positional Embedding (2D sin-cos, fixed)
        grid_size = input_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=True)

        # 3. Timestep Embedding
        # We reuse the block from your blocks.py but project it to hidden_size
        self.t_embedder = TimestepEmbedding(hidden_size)

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 5. Final Layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of conv2d)
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Zero-out adaLN modulation layers
        # This is the "Zero" in adaLN-Zero. It ensures the blocks act as identity at initialization.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = self.input_size // p
        w = self.input_size // p
        if h * w != x.shape[1]:
            raise ValueError(f"Token count mismatch in unpatchify: expected {h * w}, got {x.shape[1]}")

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t):
        """
        x: (B, C, H, W)
        t: (B,)
        """
        # 1. Patch Embedding
        # (B, C, H, W) -> (B, Hidden, H/P, W/P) -> (B, Hidden, N) -> (B, N, Hidden)
        x = self.x_embedder(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # 2. Add Positional Embedding
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)

        # 3. Timestep Embedding
        t = self.t_embedder(t) # (B, Hidden)

        # 4. Transformer Blocks
        for block in self.blocks:
            x = block(x, t)

        # 5. Final Layer
        x = self.final_layer(x, t) # (B, N, P*P*C)
        
        # 6. Unpatchify
        x = self.unpatchify(x) # (B, C, H, W)

        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return torch.from_numpy(emb).float()


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    pos = pos.reshape(-1)

    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb