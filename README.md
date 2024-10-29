# MeME
Multisubject Face Memory EEG dataset

The Epoch dataset is available at https://osf.io/dhuvs/.

To get started with the dataset, see the description below.


# Description
## Data format
The data is in standard mne Epoch file format .fif.
The data were preprocessed in MATLAB. This included a 0.2 ~ 80 Hz band-pass filter and a 45~55 Hz notch filter. 
Each epoch compromises 1000 ms, including 200 ms of baseline activity and 800 ms of post-stimulus recording.
