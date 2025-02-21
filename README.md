# Working on BraTS2020 Dataset

## Setup

### 1. Create a Python Virtual Environment

Using a virtual environment helps keep your project's dependencies isolated. Follow the steps below based on your
operating system:

#### For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### For Linux/MacOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Project Requirements

After activating your virtual environment, install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Download & Preparation

The [BraTS2020 dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation/data) can be
downloaded and prepared using the provided script:

```
python data_preparation.py
```

> **Note:** Ensure that you have the necessary API key configured to access the dataset. Place your `kaggle.json`
> credentials file in the appropriate location:
> - **Unix-based systems:**  `~/.kaggle/kaggle.json`
> - **Windows:**  `%USERPROFILE%\.kaggle\kaggle.json`

### Dataset Preprocessing Steps

The `data_preparation.py` script performs the following:

1. Downloads the BraTS2020 dataset from Kaggle.
2. Renames the segmentation mask of case 355 to match the standard format.
3. Combines the `t1ce`, `flair`, and `t2` modalities into a single NumPy array and saves it to `images/` if mask
   volume > 1% of total volume.
4. Converts corresponding masks to one-hot encoded format and saves them to `masks/`.
5. Splits the dataset into `train` (80%) and `val` (20%) directories.

## Dataset Visualization

To visualize the dataset, use the Jupyter notebook:

```
jupyter notebook notebooks/dataset-visualization.ipynb
```

This notebook:

- Plots the different MRI modalities alongside their ground truth segmentation masks.
- Demonstrates the step-by-step transformation of raw `.nii` images into NumPy arrays using `nibabel`.
- Helps to visualize how the `dataset_preparation.py` script performs the conversions.

## Post-Training Visualization

To evaluate model performance, use the visualization notebook:

```
jupyter notebook notebooks/visualize-preds.ipynb
```

This notebook:

- Analyzes metrics stored in `results/`
- Computes accuracy and Dice scores on the validation dataset
- Plots model predictions for visual inspection

## License

This project is under the MIT License. For more details click [here](LICENSE).
