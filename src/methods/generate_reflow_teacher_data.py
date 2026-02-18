import os
import argparse
import torch
import yaml
from tqdm import tqdm
from src.models import create_model_from_config
from src.methods import FlowMatching
from src.utils import EMA 
from src.data import unnormalize, save_image

def load_teacher(checkpoint_path: str, config_path: str, device: torch.device):
    print(f"Loading teacher config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading teacher weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. Create Model
    model = create_model_from_config(config).to(device)
    
    # 2. Load Weights (Handle DDP wrapping if present, just in case)
    state_dict = checkpoint['model']
    # Check if 'module.' prefix exists (it shouldn't if trained with your train.py, but safe to check)
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
        
    # 3. Apply EMA (CRITICAL FIX)
    # The teacher should use the smoothed EMA weights for best quality generation
    if 'ema' in checkpoint:
        print("✓ Loading EMA weights for Teacher")
        ema = EMA(model, decay=config['training']['ema_decay'])
        ema.load_state_dict(checkpoint['ema'])
        ema.apply_shadow() # Swap model weights with EMA weights
    else:
        print("! Warning: No EMA weights found in checkpoint. Using training weights.")
    
    model.eval()
    
    # 4. Wrap in Method
    method = FlowMatching.from_config(model, config, device)
    return method, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Teacher (with EMA)
    teacher, config = load_teacher(args.checkpoint, args.config, device)
    
    # 2. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    C = config['data']['channels']
    H = config['data']['image_size']
    W = config['data']['image_size']
    
    # 3. Generation Loop
    generated_count = 0
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    print(f"Generating {args.num_samples} pairs to {args.output_dir}...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            current_batch_size = min(args.batch_size, args.num_samples - generated_count)
            
            # A. Sample Z0
            z0 = torch.randn(current_batch_size, C, H, W, device=device)
            
            # B. Solve ODE to get Z1
            # Note: teacher.sample must support 'noise' arg (we added this to flow_matching.py)
            traj = teacher.sample(
                batch_size=current_batch_size,
                image_shape=(C, H, W),
                verbose=False,
                num_steps=args.steps,
                noise=z0 
            )
            z1 = traj[-1]
            
            # === VISUALIZATION (SANITY CHECK) ===
            # Save the first batch visualization to ensure teacher is working
            if batch_idx == 0:
                vis_path = os.path.join(args.output_dir, "sanity_check_teacher.png")
                # Take first 16 images
                vis_imgs = z1[:min(16, current_batch_size)]
                # Unnormalize [-1,1] -> [0,1]
                vis_imgs = unnormalize(vis_imgs)
                # Save grid
                save_image(vis_imgs, vis_path, nrow=4)
                print(f"Saved sanity check visualization to: {vis_path}")
            # ====================================
            
            # C. Save
            # Saving as half precision (float16) saves 50% disk space if you want
            # batch_data = {'z0': z0.cpu().half(), 'z1': z1.cpu().half()} 
            batch_data = {'z0': z0.cpu(), 'z1': z1.cpu()}
            
            save_path = os.path.join(args.output_dir, f"batch_{batch_idx:05d}.pt")
            torch.save(batch_data, save_path)
            
            generated_count += current_batch_size

    print(f"Done! Saved to {args.output_dir}")

if __name__ == '__main__':
    main()