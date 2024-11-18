# MeME
Multisubject Face Memory EEG dataset

The datasets are available at https://osf.io/dhuvs/.

To get started with the dataset, please see the description below.


# Description
## Data format
### The raw data (*.mff).
The raw data were collected through EGI’s Net Station software, which saves files in Metafile Format (MFF), which can be exported to the EDF+ format, MATLAB, EEGLAB, FieldTrip, or your own custom software.

### Half-preprocessed data (*.set).
Data were half-preprocessed using a MATLAB toolbox, EEGLAB, and datasets were saved as a .set file. 
This included a 0.2 ~ 80 Hz band-pass filter and a 45~55 Hz notch filter.

### Epoched data (*.fif).
Half pre-processed datasets were epoched through MNE Python and saved as a *.fif file. 
Each epoch compromises 1000 ms, including 200 ms of baseline activity and 800 ms of post-stimulus recording.

## Usage

### Requirements

- Python 3.8 is supported.

- Pytorch >= 1.4.0.
  
- Set up python environment:
```
conda env create -f environment.yml
conda activate brainprompt
```

### Vectorizing epochs

### Visualizing ERPs

### Classification

### Image synthesis
