"""
Flow Matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .base import BaseMethod


class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000, 
        **kwargs
    ):
        super().__init__(model, device)
        # This variable now controls the scale. 
        # It comes from your config (e.g., ddpm: num_timesteps: 1000)
        self.num_timesteps = int(num_timesteps)

    def compute_loss(self, x_1: torch.Tensor, x_0: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the Flow Matching loss (Rectified Flow Matching).
        
        Assuming shape of x_1 is (B, C, H, W)
        1. Sample noise x_0 ~ N(0, I)
        2. Sample time t uniformly [0, 1]
        3. Compute x_t = t * x_1 + (1 - t) * x_0
        4. Compute target velocity v_target = x_1 - x_0
        5. Predict velocity v_pred = model(x_t, t)
        6. MSE Loss: L = ||v_pred - v_target||^2
        7. Return loss and metrics
        
        If we input x_0, we can use this model for creating coupling flows for ReFlow
        """
        batch_size = x_1.shape[0]
        
        # 1. Sample noise x_0 ~ N(0, I)
        if x_0 is None:
            x_0 = torch.randn_like(x_1) # (B, C, H, W)
        
        # 2. Sample time t uniformly [0, 1]
        t = torch.rand((batch_size,), device=self.device)
        
        # 3. Compute x_t (Linear Interpolation)
        t_reshaped = t.view(batch_size, 1, 1, 1)
        x_t = t_reshaped * x_1 + (1.0 - t_reshaped) * x_0
        
        # 4. Compute target velocity
        v_target = x_1 - x_0
        
        # 5. Predict velocity
        # We scale t from [0, 1] to [0, num_timesteps] (e.g., 0 to 1000) (for our U-Net model)
        t_scaled = t * self.num_timesteps 
        v_pred = self.model(x_t, t_scaled)
        
        # 6. MSE Loss
        loss = F.mse_loss(v_pred, v_target)
        metrics = {'loss': float(loss.item())}
        
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        verbose: bool = True,
        num_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None, # Argument for ReFlow coupling flows, not used in standard sampling
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generates samples using Euler integration.
        """
        self.eval_mode()
        device = self.device
        
        # Use default steps if not provided
        # Note: Ideally for ODE solvers, num_steps (inference) can be different 
        # from num_timesteps (training scale), but defaulting to self.num_timesteps is safe.
        if num_steps is None:
            num_steps = self.num_timesteps
            
        # 1. Start from pure noise (t=0) or provided noise for coupling flows
        if noise is not None:
            x = noise.to(device)
            # Handle batch size mismatch if necessary
            if x.shape[0] != batch_size:
                 x = x[:batch_size]
        else:
            x = torch.randn((batch_size, *image_shape), device=device)
        
        history = [x.cpu()]
        
        # Define step size dt
        dt = 1.0 / num_steps
        
        iterator = range(num_steps)
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Flow Matching Sampling", total=num_steps)
            
        # 2. Euler Integration Loop
        for i in iterator:
            # Current time t (scalar in [0, 1])
            t_value = i / num_steps
            t_batch = torch.full((batch_size,), t_value, device=device)
            
            # Predict velocity
            # FIX: Scale using self.num_timesteps
            t_scaled = t_batch * self.num_timesteps
            v_pred = self.model(x, t_scaled)
            
            # Update x: x_{next} = x + v * dt
            x = x + v_pred * dt
            
            history.append(x.cpu())
            
        # # Clamp final output
        # history[-1] = torch.clamp(history[-1], -1.0, 1.0)
        
        return history
    
    def to(self, device: torch.device) -> "FlowMatching":
        super().to(device)
        nn.Module.to(self, device)
        self.model = self.model.to(device) 
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        # Save num_timesteps so we know what scale was used for training
        state["num_timesteps"] = self.num_timesteps
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        # Supports reading from "flow_matching" block or falling back to "ddpm" block
        fm_config = config.get("flow_matching", config.get("ddpm", config))
        return cls(
            model=model,
            device=device,
            # This pulls 1000 (or whatever you set) from your yaml
            num_timesteps=fm_config.get("num_timesteps", 1000), 
        )