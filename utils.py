import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def visualize_mri_slices(
        combined_x: np.ndarray,
        mask: np.ndarray,
        seed: Optional[int] = None
) -> None:
    """
    Generalized function to visualize MRI slices from a combined input.

    Parameters:
        combined_x (np.ndarray): 4D array with shape (H, W, S, C) containing 3 MRI modalities.
        mask (np.ndarray): 3D array representing the binary mask (tumor or region of interest).
        seed (Optional[int]): Seed for random number generation to ensure reproducibility. Default is None.

    Returns:
        None: The function displays the selected MRI slices.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    assert combined_x.ndim == 4, "combined_x should be a 4D array (H, W, S, C)."
    if mask.ndim == 4:
        mask = mask.argmax(axis=-1)

    n_slices = combined_x.shape[2]
    n_slice = random.randint(0, n_slices - 1)
    images = [combined_x[:, :, n_slice, i] for i in range(combined_x.shape[-1])] + [mask[:, :, n_slice]]
    titles = ['Flair', 'T1CE', 'T2', 'Mask']

    plt.figure(figsize=(15, 10))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 2, i + 1)
        if title == 'Mask':
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
