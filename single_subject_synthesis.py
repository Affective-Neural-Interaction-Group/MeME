import os
import torch
import pickle
import numpy as np
import pandas as pd
import ast
from PIL import Image
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURATION (Updated to match your successful environment)
# =============================================================================
CSV_PATH = './regression_detected_markers.csv'
LATENT_BASE_DIR = './latent_codes' 
GENERATOR_PATH = './stylegan3-r-ffhq-1024x1024.pkl'
SAVE_DIR = './generated_images/reconstructions_weighted'

os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# 2. INITIALIZATION
# =============================================================================
print(f"Loading StyleGAN3 from {GENERATOR_PATH}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(GENERATOR_PATH, 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device).eval()

# =============================================================================
# 3. CORE PROCESSING LOGIC
# =============================================================================
def generate_weighted_image(z_codes, weights, save_path):
    # Convert to tensor and ensure float32
    z_batch = torch.from_numpy(np.stack(z_codes)).to(device).to(torch.float32)
    labels = torch.zeros([len(z_batch), G.c_dim], device=device)
    
    with torch.no_grad():
        # 1. Map all Z to W
        w_batch = G.mapping(z_batch, labels, truncation_psi=0.8)
        
        # 2. Compute Weighted Centroid in W-space
        w_weights = torch.tensor(weights, device=device).view(-1, 1, 1)
        weighted_w = torch.sum(w_batch * w_weights, dim=0, keepdim=True) / torch.sum(w_weights)
        
        # 3. Synthesize
        img = G.synthesis(weighted_w, noise_mode="const")
        
        # 4. Post-process
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img[0].cpu().numpy()
        
    img_pil = Image.fromarray(img_np, 'RGB')
    img_pil.save(save_path)

# =============================================================================
# 4. MAIN LOOP
# =============================================================================
df = pd.read_csv(CSV_PATH)
trial_columns = [c for c in df.columns if c != 'subject']

for _, row in df.iterrows():
    subject_id = str(row['subject'])
    
    for trial_col in trial_columns:
        trial_num = trial_col.replace('t', '')
        latent_dir = os.path.join(LATENT_BASE_DIR, trial_num)
        
        # Safe Parsing
        try:
            detected_data = ast.literal_eval(row[trial_col])
        except: continue
        
        z_codes, weights = [], []
        
        for item in detected_data:
            # Handle both ('label', prob) and 'label'
            if isinstance(item, tuple):
                lbl, weight = item[0], float(item[1])
            else:
                lbl, weight = item, 1.0
            
            npy_path = os.path.join(latent_dir, f"{lbl}.npy")
            if os.path.exists(npy_path):
                z = np.load(npy_path).flatten()
                z_codes.append(z)
                weights.append(weight)
        
        if len(z_codes) > 0:
            save_img_path = os.path.join(SAVE_DIR, f"{subject_id}_{trial_num}_weighted_recon.png")
            generate_weighted_image(z_codes, weights, save_img_path)
            
print(f"Done! Images saved to {SAVE_DIR}")
