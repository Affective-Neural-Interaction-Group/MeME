import os
import torch
import pickle
import numpy as np
import pandas as pd
import ast
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

=============================================================================
1. CONFIGURATION
=============================================================================
MARKERS_CSV = './regression_detected_markers.csv'
METRICS_CSV = './regression_full_metrics.csv'
LATENT_BASE_DIR = './latent_codes'
GENERATOR_PATH = './stylegan3-r-ffhq-1024x1024.pkl'
SAVE_DIR = './generated_images/multi_reconstructions'

os.makedirs(SAVE_DIR, exist_ok=True)

AGGREGATION_LEVELS = [1, 2, 4, 8, 16, 29]
TOP_K_BLEND = 5  # We will blend the Top 5 election winners to create the final face

# =============================================================================
# 2. MODEL & DATA LOADING
# =============================================================================
print(f"Loading StyleGAN3 generator...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(GENERATOR_PATH, 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device).eval()

df_markers = pd.read_csv(MARKERS_CSV)
df_markers['subject'] = df_markers['subject'].astype(str).str.strip().str.replace('.0', '', regex=False)
df_markers = df_markers.set_index('subject')

df_metrics = pd.read_csv(METRICS_CSV)
df_metrics['Subject'] = df_metrics['Subject'].astype(str).str.strip().str.replace('.0', '', regex=False)
df_metrics['Trial_Clean'] = df_metrics['Trial'].astype(str).str.replace('t', '').str.strip()

trial_columns = [c for c in df_markers.columns if c.startswith('t')]

# =============================================================================
# 3. RECONSTRUCTION LOOP (Crowd Voting)
# =============================================================================
for trial_col in tqdm(trial_columns, desc="Synthesizing Crowd-Voted Faces"):
    trial_id = trial_col.replace('t', '')
    
    trial_metrics = df_metrics[df_metrics['Trial_Clean'] == trial_id].sort_values(by='Pearson_r', ascending=False)
    subject_votes = []
    
    # 1. Collect all votes from subjects
    for _, row in trial_metrics.iterrows():
        sub_id = row['Subject']
        pearson_score = max(0.01, float(row['Pearson_r']))
        
        if sub_id in df_markers.index:
            try:
                marker_string = str(df_markers.loc[sub_id, trial_col])
                detected_data = ast.literal_eval(marker_string)
                votes = []
                for item in detected_data:
                    lbl, prob = (item[0], float(item[1])) if isinstance(item, tuple) else (item, 1.0)
                    clean_lbl = str(lbl).strip().replace('.npy', '')
                    votes.append((clean_lbl, prob * pearson_score))
                if votes:
                    subject_votes.append(votes)
            except: pass

    # 2. Process each N-level crowd size
    for N in AGGREGATION_LEVELS:
        if len(subject_votes) < N: continue
            
        subset_votes = subject_votes[:N]
        image_scoreboard = defaultdict(float)
        
        # Tally the points
        # 
        for sub_vote_list in subset_votes:
            for lbl, weighted_prob in sub_vote_list:
                image_scoreboard[lbl] += weighted_prob
                
        # 3. Get the Top K Election Winners
        top_candidates = sorted(image_scoreboard.items(), key=lambda x: x[1], reverse=True)[:TOP_K_BLEND]
        
        z_codes, weights = [], []
        for lbl, score in top_candidates:
            path = os.path.join(LATENT_BASE_DIR, trial_id, f"{lbl}.npy")
            if os.path.exists(path):
                z_codes.append(np.load(path).flatten())
                weights.append(score) # Use their total crowd vote as their blending weight
        
        # 4. Synthesize the Consensus Image
        if len(z_codes) > 0:
            z_batch = torch.from_numpy(np.stack(z_codes)).to(device).to(torch.float32)
            labels = torch.zeros([len(z_batch), G.c_dim], device=device)
            
            with torch.no_grad():
                # Map discrete winners to W-space
                w_batch = G.mapping(z_batch, labels, truncation_psi=0.7)
                
                # Blend the winners together based on how many votes they got
                w_weights = torch.tensor(weights, device=device).view(-1, 1, 1)
                w_centroid = torch.sum(w_batch * w_weights, dim=0, keepdim=True) / torch.sum(w_weights)
                
                # Render
                img = G.synthesis(w_centroid, noise_mode="const")
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_np = img[0].cpu().numpy()
            
            save_name = f"trial-{trial_id}_N-{N}_voting_recon.png"
            Image.fromarray(img_np, 'RGB').save(os.path.join(SAVE_DIR, save_name))

print(f"\nSuccess! Crowd-voted reconstructions saved to: {SAVE_DIR}")

