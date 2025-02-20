import os
from glob import glob
from os import PathLike
from typing import Optional

import numpy as np
import torch
from colorama import Fore, Style
from torch.utils.data import Dataset, DataLoader

from utils import visualize_mri_slices


class BraTS2020Dataset(Dataset):
    def __init__(
            self,
            image_dir: str | PathLike[str] = "data/input_data_total/images",
            mask_dir: str | PathLike[str] = "data/input_data_total/masks",
            verbose: Optional[bool] = False,
    ) -> None:
        super(BraTS2020Dataset, self).__init__()
        self.images = self._load_npy_files(image_dir, verbose=verbose)
        self.masks = self._load_npy_files(mask_dir, verbose=verbose)
        self.verbose = verbose

        if self.verbose:
            print(
                f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded "
                f"{Fore.YELLOW}{len(self.images)}{Style.RESET_ALL} images and "
                f"{Fore.YELLOW}{len(self.masks)}{Style.RESET_ALL} masks."
            )

            print(
                f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Image tensor shape: "
                f"{Fore.GREEN}{self.images[0].shape}{Style.RESET_ALL} | "
                f"Mask tensor shape: {Fore.GREEN}{self.masks[0].shape}{Style.RESET_ALL}"
            )

    @staticmethod
    def _load_npy_files(directory: str | PathLike[str], verbose: Optional[bool] = False) -> list[torch.Tensor]:
        """Helper function to load all .npy files from a directory into a numpy array."""
        file_paths = sorted(glob(os.path.join(directory, "*.npy")))

        if not file_paths:
            raise FileNotFoundError(f"[ERROR] No .npy files found in {directory}")

        if verbose:
            print(
                f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Found "
                f"{Fore.YELLOW}{len(file_paths)}{Style.RESET_ALL} .npy files in {directory}"
            )

        return [torch.from_numpy(np.load(path)).type(torch.float32).permute(3, 0, 1, 2) for path in file_paths]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]


def test():
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Running dataset test...")

    dataset = BraTS2020Dataset()
    image_tensor, mask_tensor = dataset[0]

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} First image dtype: {Fore.MAGENTA}{image_tensor.dtype}{Style.RESET_ALL}, "
        f"shape: {Fore.GREEN}{image_tensor.shape}{Style.RESET_ALL}"
    )

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} First mask dtype: {Fore.MAGENTA}{mask_tensor.dtype}{Style.RESET_ALL}, "
        f"shape: {Fore.GREEN}{mask_tensor.shape}{Style.RESET_ALL}"
    )

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Dataset size: {Fore.YELLOW}{len(dataset)}{Style.RESET_ALL}"
    )

    visualize_mri_slices(image_tensor, mask_tensor, seed=42)

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Dataset test completed successfully. Creating DataLoader...")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=os.cpu_count())

    batch = next(iter(dataloader))

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded batch of "
        f"{Fore.YELLOW}{len(batch[0])}{Style.RESET_ALL} samples with batch size "
        f"{Fore.YELLOW}4{Style.RESET_ALL}, shuffle={Fore.YELLOW}True{Style.RESET_ALL}."
    )

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Batch image shape: {Fore.GREEN}{batch[0].shape}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Batch mask shape: {Fore.GREEN}{batch[1].shape}{Style.RESET_ALL}")


if __name__ == "__main__":
    test()
