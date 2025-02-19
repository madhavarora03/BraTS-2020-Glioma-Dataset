import glob
import os
from os import PathLike

import nibabel as nib
import numpy as np
import splitfolders
from colorama import Fore, Style, init
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils import to_categorical

# Initialize colorama
init(autoreset=True)

# Initialize MinMaxScaler
scaler = MinMaxScaler()


def get_file_lists(data_dir: str | PathLike[str]="data/training_data/"):
    """ Get sorted lists of file paths for different modalities and masks. """
    t2_list = sorted(glob.glob(f"{data_dir}/**/*t2.nii"))
    t1ce_list = sorted(glob.glob(f"{data_dir}/**/*t1ce.nii"))
    flair_list = sorted(glob.glob(f"{data_dir}/**/*flair.nii"))
    mask_list = sorted(glob.glob(f"{data_dir}/**/*seg.nii"))

    assert len(t2_list) == len(t1ce_list) == len(flair_list) == len(mask_list), (
        f"{Fore.RED}❌ Length mismatch{Style.RESET_ALL}: All modality lists and mask lists should have the same number of items."
    )

    print(f"✔ Found {Fore.GREEN}{len(t2_list)}{Style.RESET_ALL} samples.")
    return t2_list, t1ce_list, flair_list, mask_list


def load_and_preprocess_image(image_path: str | PathLike[str]):
    """ Load a NIfTI image and apply MinMax scaling. """
    image = nib.load(image_path).get_fdata()
    return scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)


def load_and_process_mask(mask_path: str | PathLike[str]):
    """ Load a NIfTI mask, convert to uint8, and adjust class labels. """
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    mask[mask == 4] = 3  # Convert class 4 to class 3
    return mask


def process_and_save_images(t2_list, t1ce_list, flair_list, mask_list, output_dir: str | PathLike[str]="data/input_data_total") -> None:
    """ Process and save images & masks if they contain enough tumor pixels. """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for img_idx in tqdm(range(len(t2_list)), desc="Processing images", colour="green"):
        print(f"⚙️ Now processing {Fore.BLUE}image and mask number: {img_idx}{Style.RESET_ALL}")

        # Load and preprocess modalities
        t2_img = load_and_preprocess_image(t2_list[img_idx])
        t1ce_img = load_and_preprocess_image(t1ce_list[img_idx])
        flair_img = load_and_preprocess_image(flair_list[img_idx])

        # Load and process mask
        mask_img = load_and_process_mask(mask_list[img_idx])

        # Crop region of interest
        combined_img = np.stack([flair_img, t1ce_img, t2_img], axis=3)[56:184, 56:184, 13:141]
        mask_img = mask_img[56:184, 56:184, 13:141]

        # Calculate class distribution in mask
        val, counts = np.unique(mask_img, return_counts=True)
        tumor_ratio = 1 - (counts[0] / counts.sum())

        if tumor_ratio > 0.01:  # At least 1% useful volume with labels that are not 0
            print(f"{Fore.GREEN}✔ Saving{Style.RESET_ALL}: Significant tumor region found.")
            mask_img = to_categorical(mask_img, num_classes=4)
            np.save(f"{output_dir}/images/image_{img_idx}.npy", combined_img)
            np.save(f"{output_dir}/masks/mask_{img_idx}.npy", mask_img)
        else:
            print(f"{Fore.RED}❌ Skipping{Style.RESET_ALL}: Insufficient tumor presence.")


# Run the pipeline
t2_files, t1ce_files, flair_files, mask_files = get_file_lists()
process_and_save_images(t2_files, t1ce_files, flair_files, mask_files)

# Split data with a ratio into train and test.
input_path = "data/input_data_total"
output_path = "data/input_data_split"

splitfolders.ratio(input_path, output_path, seed=42, ratio=(0.75, 0.25), group_prefix=None)

# Counting the number of images in the train and val directories
print(Fore.GREEN + f"There are {len(os.listdir(os.path.join(output_path, 'train/images')))} images in train.")
print(Fore.BLUE + f"There are {len(os.listdir(os.path.join(output_path, 'val/images')))} images in val.")