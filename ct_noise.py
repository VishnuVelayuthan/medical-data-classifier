import os
from typing import List

import nibabel as nib
import numpy as np


class CTScanDataset:
    """Simple dataset interface for a directory of .nii.gz CT scans."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_paths: List[str] = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nii.gz")
        ]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> nib.Nifti1Image:
        return nib.load(self.file_paths[idx])


def add_gaussian_noise(scan: nib.Nifti1Image, std: float = 25.0) -> nib.Nifti1Image:
    """Add Gaussian noise to the voxel intensities of a CT scan."""
    data = scan.get_fdata()
    noise = np.random.normal(0.0, std, size=data.shape)
    noisy_data = data + noise
    return nib.Nifti1Image(noisy_data, scan.affine, scan.header)


def save_scan(scan: nib.Nifti1Image, path: str) -> None:
    nib.save(scan, path)


def augment_dataset_with_noise(dataset: CTScanDataset, output_dir: str, std: float = 25.0) -> None:
    """Apply random Gaussian noise to each scan in the dataset and save the results."""
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(len(dataset)):
        scan = dataset[idx]
        noisy_scan = add_gaussian_noise(scan, std=std)
        filename = os.path.basename(dataset.file_paths[idx])
        out_path = os.path.join(output_dir, f"noisy_{filename}")
        save_scan(noisy_scan, out_path)


__all__ = [
    "CTScanDataset",
    "add_gaussian_noise",
    "save_scan",
    "augment_dataset_with_noise",
]

