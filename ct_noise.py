import os
from typing import List

import nibabel as nib
import numpy as np
import argparse

from monai.transforms import RandGaussianNoise


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


def add_gaussian_noise(scan: nib.Nifti1Image, std: float = 25) -> nib.Nifti1Image:
    """Add Gaussian noise to the voxel intensities of a CT scan."""
    data = scan.get_fdata()
    noisy_data = RandGaussianNoise(prob=1.0, mean=0.0, std=std)(data)  
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
        print(f"Created noisy scan for {dataset.file_paths[idx]} with std={std}")

if __name__ == "__main__": 


    parser = argparse.ArgumentParser(description="Augment CT scans with Gaussian noise.")
    parser.add_argument("data_dir", type=str, help="Directory containing .nii.gz CT scans.")
    parser.add_argument("output_dir", type=str, help="Directory to save noisy scans.")
    parser.add_argument("--std", type=float, default=1000.0, help="Standard deviation of Gaussian noise.")

    args = parser.parse_args()

    dataset = CTScanDataset(args.data_dir)
    augment_dataset_with_noise(dataset, args.output_dir, std=args.std)

