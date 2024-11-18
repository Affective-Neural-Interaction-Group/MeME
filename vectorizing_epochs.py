'''Download the epoch data ends with .fif from the following link:  https://osf.io/dhuvs/'''

import os
import numpy as np
import mne
from mne import Epochs

# vectorize epochs function
def vectorize_epochs(epochs):
    start, stop = epochs.time_as_index([0.05, 0.8])  # Convert 50ms and 800ms to indices
    n_windows = 15
    window_length = (stop - start) // n_windows

    # Initialize an empty array to hold the averaged data
    n_epochs, n_channels = len(epochs), len(epochs.ch_names)
    X = np.zeros((n_epochs, n_channels, n_windows))

    # Loop through each epoch and channel to calculate the averages
    for i, epoch in enumerate(epochs.get_data()):
        for j in range(n_channels):
            for k in range(n_windows):
                window_start = start + k * window_length
                window_stop = start + (k + 1) * window_length
                X[i, j, k] = epoch[j, window_start:window_stop].mean()

    return X
        
# Load the epoch data
epochs_dir = 'Enter the path to the folder where the epoch data is stored'
files = os.listdir(epochs_dir)

# Loop through the files and load the epoch data
for file in files:
    if file.endswith('.fif'):
        epochs = mne.read_epochs(os.path.join(epochs_dir, file))
        
        # Vectorize the epochs data
        data = epochs.get_data()
        X = vectorize_epochs(epochs)
        
        # Save the vectorized data
        np.save(os.path.join(epochs_dir, file.replace('.fif', '_vectorized.npy')), X)