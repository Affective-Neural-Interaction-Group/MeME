import mne
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. ENHANCED ARCHITECTURE (Non-Linear Mapping)
# =============================================================================
class EEGDeepRegressor(nn.Module):
    def __init__(self, input_dim):
        super(EEGDeepRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def train_model(X_train, y_train, input_dim, epochs=100, seed=42):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGDeepRegressor(input_dim).to(device)
    
    # Huber Loss (SmoothL1) handles noisy EEG outliers better than MSE
    criterion = nn.SmoothL1Loss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32, device=device), 
                            torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model, device

# =============================================================================
# 2. DATA PREPARATION (P300 Windowing)
# =============================================================================
def load_and_window_data(fif_path, latent_base_dir):
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    
    # METHODOLOGICAL REVISION: Focus on P300 window (200-600ms)
    # This removes visual processing noise and late-stage cognitive artifacts
    epochs.crop(tmin=0.2, tmax=0.6)
    
    epoch_dict = {
        '1010': epochs[:255], '355':  epochs[255:510], '1172': epochs[510:765],
        '1111': epochs[765:1020], '1156': epochs[1020:1275], '1223': epochs[1275:1530],
        '193':  epochs[1530:1785], '209':  epochs[1785:2040], '439':  epochs[2040:2295],
        '908':  epochs[2295:2550], '587':  epochs[2550:2805], '2220': epochs[2805:3060],
        '6251': epochs[3060:3315], '269':  epochs[3315:3570], '1160': epochs[3570:3825]
    }
    
    subject_data = {}
    for trial_num, cur_epochs in epoch_dict.items():
        if len(cur_epochs) == 0: continue
        
        trial_dir = os.path.join(latent_base_dir, trial_num)
        target_vec = np.load(os.path.join(trial_dir, "t000.npy")).flatten()
        
        rev_id = {v: k for k, v in cur_epochs.event_id.items()}
        labels = [rev_id[e[2]] for e in cur_epochs.events]
        
        valid_idx, dists = [], []
        for idx, lbl in enumerate(labels):
            path = os.path.join(trial_dir, f"{lbl}.npy")
            if os.path.exists(path):
                valid_idx.append(idx)
                dists.append(np.linalg.norm(np.load(path).flatten() - target_vec))
                
        if not valid_idx: continue
        X_eeg = cur_epochs.get_data(copy=True)[valid_idx]
        subject_data[trial_num] = (X_eeg, np.array(dists), np.array(labels)[valid_idx])
        
    return subject_data

# =============================================================================
# 3. LOSO MAIN LOOP
# =============================================================================
EPOCHS_DIR = './epochs'
LATENT_BASE_DIR = './latent_codes'
TOP_K = 15

files = [f for f in os.listdir(EPOCHS_DIR) if f.endswith('.fif')]
final_markers = {}

for file in files:
    sub_id = file.split('.')[0]
    print(f"Training Enhanced Regressor for Subject: {sub_id}")
    
    data_dict = load_and_window_data(os.path.join(EPOCHS_DIR, file), LATENT_BASE_DIR)
    trial_keys = list(data_dict.keys())
    input_dim = data_dict[trial_keys[0]][0].shape[1] * data_dict[trial_keys[0]][0].shape[2]
    sub_markers = {}

    for test_trial in trial_keys:
        # Split
        X_test, y_test_raw, test_labels = data_dict[test_trial]
        X_train = np.concatenate([data_dict[k][0] for k in trial_keys if k != test_trial])
        y_train_raw = np.concatenate([data_dict[k][1] for k in trial_keys if k != test_trial])

        # Feature Scaling
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
        X_test = scaler_X.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape)

        # Target Scaling [0, 1]
        scaler_y = MinMaxScaler().fit(y_train_raw.reshape(-1, 1))
        y_train_scaled = scaler_y.transform(y_train_raw.reshape(-1, 1)).flatten()

        # Train Deep model
        model, device = train_model(X_train, y_train_scaled, input_dim, epochs=100)
        
        # Predict & Sort
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy().flatten()
        
        # Extract Top-K closest images (lowest predicted distance)
        top_idx = np.argsort(preds)[:TOP_K]
        
        # Format weights for centroid script (inverse distance)
        sub_markers[f"t{test_trial}"] = [(test_labels[i], float(1.0 - preds[i])) for i in top_idx]

    final_markers[sub_id] = sub_markers

pd.DataFrame.from_dict(final_markers, orient='index').to_csv('regression_detected_markers.csv')
print("Markers saved to regression_detected_markers.csv")
