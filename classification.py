import mne
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
from imblearn.over_sampling import SMOTE

# Configuration
epochs_dir = '/home/yc47480/brainprompt/epochs'
n_folds = 5
n_csp_components = 6  # Increased from 4 to capture more detail

files = os.listdir(epochs_dir)
detected_markers = {}
performance_records = []

# Initialize CSP strictly for feature extraction
# We fit CSP on training data only inside the loop to avoid leakage
csp = CSP(n_components=n_csp_components, reg='ledoit_wolf', log=True, norm_trace=False)

# Initialize LDA with Shrinkage (Better for high-dimensional EEG)
lda = LDA(solver='lsqr', shrinkage='auto')

for file in files:
    if file.endswith('.fif'):
        print(f"\nProcessing Subject: {file}")
        filename = file.split('.')[0]
        epochs = mne.read_epochs(os.path.join(epochs_dir, file), preload=True, verbose=False)

        # 0. Pre-processing: Drop bad epochs (Artifact Rejection)
        # Drop epochs with variance > 150uV (blinks/movement)
        epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False)

        # --- Data Slicing ---
        epoch_dict = {
            't1010': epochs[:255],
            't355':  epochs[255:510],
            't1172': epochs[510:765],
            't1111': epochs[765:1020],
            't1156': epochs[1020:1275],
            't1223': epochs[1275:1530],
            't193':  epochs[1530:1785],
            't209':  epochs[1785:2040],
            't439':  epochs[2040:2295],
            't908':  epochs[2295:2550],
            't587':  epochs[2550:2805],
            't2220': epochs[2805:3060],
            't6251': epochs[3060:3315],
            't269':  epochs[3315:3570],
            't1160': epochs[3570:3825]
        }

        trial_mark = {}

        for epoch_name, cur_epochs in epoch_dict.items():
            
            # 1. Extract Target (t) and Non-Target (nt)
            t_ids = [k for k in cur_epochs.event_id if k.startswith('t')]
            nt_ids = [k for k in cur_epochs.event_id if not k.startswith('t')]

            if not t_ids or not nt_ids:
                continue

            t_epochs = cur_epochs[t_ids]
            nt_epochs = cur_epochs[nt_ids]

            # Get Data (Epochs x Channels x Time)
            Xt = t_epochs.get_data(copy=True)
            Xnt = nt_epochs.get_data(copy=True)

            # Skip blocks with too few targets (SMOTE needs at least ~6 samples)
            if len(Xt) < 6:
                print(f"  [{epoch_name}] Skipped: Not enough targets for CV.")
                continue

            # 2. Prepare Labels and IDs
            # We use ALL non-targets now (No undersampling yet)
            X_raw = np.concatenate((Xt, Xnt), axis=0)
            
            # Create labels: 1 = Target, 0 = Non-Target
            y_raw = np.concatenate((np.ones(len(Xt)), np.zeros(len(Xnt))))

            # Store marker strings for reconstruction
            rev_event_id = {v: k for k, v in cur_epochs.event_id.items()}
            yt_labels = [rev_event_id[e[2]] for e in t_epochs.events]
            ynt_labels = [rev_event_id[e[2]] for e in nt_epochs.events]
            raw_marker_labels = np.array(yt_labels + ynt_labels)

            # 3. Cross-Validation with SMOTE Inside the Loop
            # Critical: SMOTE must only be applied to TRAINING data, never test data.
            
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            # Arrays to store results
            y_true_all = []
            y_pred_all = []
            y_prob_all = []
            test_indices_all = []

            for train_idx, test_idx in cv.split(X_raw, y_raw):
                # Split Train/Test
                X_train_3d, X_test_3d = X_raw[train_idx], X_raw[test_idx]
                y_train, y_test = y_raw[train_idx], y_raw[test_idx]

                # A. Fit CSP on Training Data Only
                # Transform 3D EEG -> 2D Features
                X_train_csp = csp.fit_transform(X_train_3d, y_train)
                X_test_csp = csp.transform(X_test_3d)

                # B. Apply SMOTE to Training Features Only
                # This balances the class 1 vs class 0 count by creating synthetic targets
                smote = SMOTE(random_state=42)
                X_train_bal, y_train_bal = smote.fit_resample(X_train_csp, y_train)

                # C. Train LDA on Balanced Data
                lda.fit(X_train_bal, y_train_bal)

                # D. Predict on (Imbalanced) Test Data
                y_pred = lda.predict(X_test_csp)
                y_prob = lda.predict_proba(X_test_csp)[:, 1]

                # Store results
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
                y_prob_all.extend(y_prob)
                test_indices_all.extend(test_idx)

            # 4. Calculate Metrics
            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)
            y_prob_all = np.array(y_prob_all)
            test_indices_all = np.array(test_indices_all)

            acc = accuracy_score(y_true_all, y_pred_all)
            auc = roc_auc_score(y_true_all, y_prob_all)
            f1 = f1_score(y_true_all, y_pred_all)

            print(f"  [{epoch_name:<6}] AUC: {auc:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")

            performance_records.append({
                'Subject': filename,
                'Trial_Block': epoch_name,
                'AUC': auc,
                'Accuracy': acc,
                'F1': f1
            })

            # 5. Retrieve Markers (Reconstruction Logic)
            # Map predictions back to the original file indices
            # We identify which original indices were predicted as '1' (Target)
            
            predicted_as_target_mask = (y_pred_all == 1)
            original_indices_predicted_target = test_indices_all[predicted_as_target_mask]
            
            # Retrieve the string labels
            retrieved_markers = raw_marker_labels[original_indices_predicted_target].tolist()
            trial_mark[epoch_name] = retrieved_markers

        detected_markers[filename] = trial_mark

# Save Results
df_markers = pd.DataFrame.from_dict(detected_markers, orient='index')
df_markers.to_csv('detected_markers_smote.csv')

df_perf = pd.DataFrame(performance_records)
df_perf.to_csv('performance_metrics_smote.csv', index=False)

print("\nProcessing complete.")
if not df_perf.empty:
    print(f"Average AUC: {df_perf['AUC'].mean():.4f}")
