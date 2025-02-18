# Working on BraTS2020 Dataset

## Setup

### 1. Create a Python Virtual Environment

Using a virtual environment helps keep your project's dependencies isolated. Follow the steps below based on your operating system:

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

## Dataset Download

To download the BraTS2020 dataset, run the provided download script. This script uses the `kagglehub` package to automatically download the latest dataset version.

Run the script with:

```bash
python dataset_download.py
```

> **Note:** Ensure that you have the necessary permissions and API keys configured to access the dataset. To do so, place your `kaggle.json` credentials file in the appropriate location:  
> - **For Unix-based systems:** `~/.kaggle/kaggle.json`  
> - **For Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
