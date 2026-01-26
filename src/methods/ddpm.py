"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
        schedule_type: str = "linear",
        parameterization: str = "epsilon", # Options: "epsilon", "x0"
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        # TODO: Implement your own init
        self.schedule_type = schedule_type
        self.parameterization = parameterization

        # =========================================================================
        # You can add, delete or modify as many functions as you would like
        # =========================================================================
        
        # Pro tips: If you have a lot of pseudo parameters that you will specify for each
        # model run but will be fixed once you specified them (say in your config),
        # then you can use super().register_buffer(...) for these parameters

        # Pro tips 2: If you need a specific broadcasting for your tensors,
        # it's a good idea to write a general helper function for that

        # 1. Get beta schedule
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule_type == "quadratic":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # 2. Precompute alphas and other useful terms for sampling and training
        alphas = 1.0 - betas # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0) # (T,) (product of alphas from 1 to t)
        # In this line, we compute alpha_t-1 used in posterior q(x_{t-1} | x_t, x_0)'s mean and variance
        # We remove the last element of alphas_cumprod and pad a 1.0 at the beginning
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # (T,)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # (T,)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod) # (T,)
        
        # 3. Precompute terms for the reverse process i.e. posterior q(x_{t-1} | x_t, x_0)
        sqrt_reciprocal_alphas = torch.sqrt(1.0 / alphas) # (T,)
        
        # computing posterior variance i.e. beta_tilde for q(x_{t-1} | x_t, x_0) all timesteps
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) # (T,)
        posterior_variance_clamped = torch.max(posterior_variance, torch.tensor(1e-20)) # (T,) # to prevent numerical issues with log(0)
        posterior_log_variance_clipped = torch.log(posterior_variance_clamped) # (T,)
        
        # ALTERNATE PARAMETERIZATION:
        # In the DDPM paper, they also mention an alternate parameterization of the reverse process
        # where they predict x_0 directly instead of the noise. In that case, the mean of the posterior is computed differently.
        # Coef 1: Multiplies x_0
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Coef 2: Multiplies x_t
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        
        # 4. Register buffers so that they are saved and moved to the correct device automatically
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_reciprocal_alphas", sqrt_reciprocal_alphas)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
    
    # Helper function to extract value and broadcast it to the shape of x
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Helper function to extract values from a 1-D tensor `a` at indices `t` and
        reshape to `x_shape` for broadcasting.
        
        Args:
            a: 1-D tensor of constants (e.g., betas)
            t: 1-D tensor of time indices
            x_shape: Shape of the input tensor (B, C, H, W)
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    
    # =========================================================================
    # Forward process
    # =========================================================================
    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:                
        # TODO: Implement the forward (noise adding) process of DDPM
        # raise NotImplementedError
        """
        Diffuses the data x_0 to time t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0) # Sample noise if not provided # (B, C, H, W)
        
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape) # (B, 1, 1, 1) # we get sqrt(alpha_bar_t)
        sqrt_alpha_one_minus_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) # (B, 1, 1, 1) # we get sqrt(1 - alpha_bar_t)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_alpha_one_minus_cumprod_t * noise
        

    # =========================================================================
    # Training loss
    # =========================================================================
    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here
        Computes simple MSE loss between the added noise and the predicted noise by the model.

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width) i.e. (B, C, H, W)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        batch_size = x_0.shape[0]
        
        # 1. Sample random time steps for each sample in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device).long() # (B,)
        
        # 2. Sample random noise from standard normal distribution
        noise = torch.randn_like(x_0) # (B, C, H, W)
        
        # 3. Compute noisy image i.e. x_t using forward process
        x_t = self.forward_process(x_0, t, noise) # (B, C, H, W)
        
        # 4. Get the model output (might be noise or x_0 depending on parameterization)
        model_output = self.model(x_t, t) # (B, C, H, W)
        
        # 5. Get the target depending on parameterization
        if self.parameterization == "epsilon":
            # TARGET IS NOISE
            target = noise
        elif self.parameterization == "x0":
            # TARGET IS x_0
            target = x_0
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        
        # 5. Compute MSE loss between the true noise and the predicted noise
        # loss = F.mse_loss(noise_pred, noise, reduction='mean')
        loss = F.mse_loss(model_output, target, reduction='mean')
        metrics = {'loss': float(loss.item())}
        return loss, metrics
        # raise NotImplementedError

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process
        # Performs one step of the reverse diffusion process to obtain x_{t-1} from x_t.

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        # raise NotImplementedError
        batch_size = x_t.shape[0]
        
        # 1. Predict the model output (noise or x_0 depending on parameterization)
        model_output = self.model(x_t, t) # (B, C, H, W)
        
        # 2. Compute the mean of the posterior q(x_{t-1} | x_t, x_0) (based on parameterization)
        if self.parameterization == "epsilon":
            # mu = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * noise_pred)
            sqrt_reciprocal_alpha_t = self._extract(self.sqrt_reciprocal_alphas, t, x_t.shape) # (B, 1, 1, 1)
            betas_t = self._extract(self.betas, t, x_t.shape) # (B, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) # (B, 1, 1, 1)
            
            mu = sqrt_reciprocal_alpha_t * (x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * model_output) # (B, C, H, W)
            
        elif self.parameterization == "x0":
            # New formula: Posterior mean using predicted x_0
            pred_x0 = model_output # (B, C, H, W)
            
            # # Optional: Clip pred_x0 to [-1, 1] for stability (highly recommended for x0 prediction)
            # pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Get coefficients
            coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
            coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
            
            # mu = coef1 * x_0 + coef2 * x_t
            mu = coef1 * pred_x0 + coef2 * x_t
            
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        
        # 3. Add noise i.e. variance
        # x_{t-1} = mu + sqrt(posterior_variance_t) * z, where z ~ N(0, I)
        # We only add noise if t > 0
        mask = (t > 0).float().view(batch_size, 1, 1, 1) # (B, 1, 1, 1)
        
        # The DDPM paper gives two options for variance:
        # Option 1: Use a fixed variance sigma_t^2 = beta_t
        # Option 2: Use the posterior variance computed during initialization (we use this option here for stability)
        posterior_log_var_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape) # (B, 1, 1, 1)
        noise = torch.randn_like(x_t) # (B, C, H, W)
        
        # x_{t-1} = mu + sqrt(variance) * z
        x_prev = mu + mask * torch.exp(0.5 * posterior_log_var_t)* noise # (B, C, H, W)
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        verbose: bool = True,
        num_steps: Optional[int] = 1000,
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()
        Generates samples by running the full reverse process from T to 0

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        # raise NotImplementedError
        device = self.device
        # 1. Sample pure noise to start sampling i.e. x_T
        x_t = torch.randn((batch_size, *image_shape), device=device) # (B, C, H, W)
        
        # Store history: [x_T, x_T-1, ..., x_0]
        history = [x_t.cpu()]
        
        # 2. Iteratively apply reverse_process from t = T-1 to t = 0
        iterator = range(num_steps - 1, -1, -1)
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="DDPM Sampling", total=num_steps)
            
        for t in iterator:
            timestep_batch_size = torch.full((batch_size,), t, device=device, dtype=torch.long) # (B,) # create a batch of the same timestep
            x_t_1 = self.reverse_process(x_t, timestep_batch_size) # (B, C, H, W)
            history.append(x_t_1.cpu())
            # Update x_t for the next iteration
            x_t = x_t_1
        
        # Return the trajectory of samples from x_T to x_0
        # Take the last element as the final samples
        # # Ensure the final image is clamped
        # if self.parameterization == "x0":
        #     history[-1] = torch.clamp(history[-1], -1.0, 1.0)
        return history
    
    # =========================================================================
    # Device / state
    # =========================================================================

    # def to(self, device: torch.device) -> "DDPM":
    #     super().to(device)
    #     self.device = device
    #     return self
    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        nn.Module.to(self, device)
        self.model = self.model.to(device) 
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        # TODO: add other things you want to save
        state["parameterization"] = self.parameterization
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
            parameterization=ddpm_config.get("parameterization", "epsilon")
        )
