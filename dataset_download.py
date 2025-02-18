import os
import shutil

import kagglehub
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Download the latest dataset version
path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
print(f"{Fore.GREEN}Path to dataset files:{Style.RESET_ALL} {path}")

# Check if 'data' directory exists, if not, copy the dataset
if not os.path.exists("data"):
    try:
        shutil.copytree(f"{path}/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData", "data/")
        print(f"{Fore.CYAN}Dataset copied to 'data/' directory.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error copying dataset:{Style.RESET_ALL} {e}")

# Check if the specific problematic file exists and rename it if necessary
wrong_file_path = "data/BraTS20_Training_355/W39_1998.09.19_Segm.nii"
corrected_file_path = "data/BraTS20_Training_355/BraTS20_Training_355_seg.nii"

if os.path.exists(wrong_file_path):
    try:
        os.rename(wrong_file_path, corrected_file_path)
        print(f"{Fore.YELLOW}Renamed faulty file into correct format:{Style.RESET_ALL} {corrected_file_path}")
    except Exception as e:
        print(f"{Fore.RED}Error renaming file:{Style.RESET_ALL} {e}")
else:
    print(f"{Fore.RED}File not found:{Style.RESET_ALL} {wrong_file_path}")
