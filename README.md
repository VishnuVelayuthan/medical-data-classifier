# medical-data-classifier

This repository contains utilities for handling CT scan datasets stored as `.nii.gz` files.

## CT Noise Augmentation

The `ct_noise.py` module provides a simple interface for loading CT scans, adding Gaussian noise, and saving the noisy scans back to disk.

Example usage:

```python
from ct_noise import CTScanDataset, augment_dataset_with_noise

dataset = CTScanDataset("/path/to/ct_scans")
augment_dataset_with_noise(dataset, output_dir="noisy_scans", std=25.0)
```

This will create a `noisy_scans/` folder containing new `.nii.gz` files with added noise.
