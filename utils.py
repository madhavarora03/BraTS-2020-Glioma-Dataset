import random
from glob import glob
from typing import Optional, TypedDict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def visualize_mri_slices(
        combined_x: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        save_path: Optional[str] = None,
        seed: Optional[int] = None
) -> None:
    """
    Generalized function to visualize MRI slices from a combined input.

    Parameters:
        combined_x (np.ndarray | torch.Tensor): 4D array with shape (C, H, W, S) containing 3 MRI modalities.
        mask (np.ndarray | torch.Tensor): 3D or 4D array representing the binary mask (tumor or region of interest).
        save_path (Optional[str]): path to save the visualization.
        seed (Optional[int]): Seed for random number generation to ensure reproducibility.

    Returns:
        None: Displays the selected MRI slices.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Convert to NumPy if tensor
    if isinstance(combined_x, torch.Tensor):
        combined_x = combined_x.detach().cpu().numpy()
        combined_x = np.transpose(combined_x, (1, 2, 3, 0))
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
        mask = np.transpose(mask, (1, 2, 3, 0))

    if mask.ndim == 4:
        mask = mask.argmax(axis=-1)  # Convert one-hot mask to categorical labels

    assert combined_x.ndim == 4, "combined_x should be a 4D array (H, W, S, C)."

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

    if save_path is not None:
        plt.savefig(save_path)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def save_checkpoint(state, filename="checkpoint.pth", trace_fn=print, verbose=True):
    """ Save checkpoint """
    if verbose:
        trace_fn("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, trace_fn=print):
    """ Load checkpoint """
    trace_fn("=> Loading checkpoint")
    model.load_state_dict(torch.load(checkpoint))


def calc_dice_score(preds: torch.Tensor, targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None, eps=1e-8):
    """
    Compute the Weighted Dice Score for multi-class segmentation.

    Args:
        preds (torch.Tensor): Model predictions (batch_size, num_classes, D, H, W), logits or probabilities.
        targets (torch.Tensor): Ground truth masks (batch_size, num_classes, D, H, W), one-hot encoded.
        class_weights (Optional[torch.Tensor]): Weights for each class (num_classes,).
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Weighted Dice Score over all classes.
    """
    num_classes = preds.shape[1]

    preds = torch.softmax(preds, dim=1)  # Convert logits to probabilities

    if class_weights is None:
        class_weights = torch.ones(num_classes, device=preds.device) / num_classes
    else:
        class_weights = class_weights / class_weights.sum()  # Normalize weights

    dice = torch.zeros(num_classes, device=preds.device)

    for c in range(num_classes):
        pred_c = preds[:, c, :, :, :]
        target_c = targets[:, c, :, :, :]  # Directly use the one-hot encoded targets

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice[c] = class_weights[c] * (2. * intersection + eps) / (union + eps)

    return dice.sum().item()


def calc_accuracy(preds: torch.Tensor, targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None,
                  eps=1e-8) -> float:
    """
    Compute the Weighted Accuracy for multi-class segmentation.

    Args:
        preds (torch.Tensor): Model predictions (batch_size, num_classes, D, H, W).
        targets (torch.Tensor): Ground truth masks (batch_size, num_classes, D, H, W), one-hot encoded.
        class_weights (Optional[torch.Tensor]): Weights for each class.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Weighted Accuracy over all classes.
    """
    num_classes = preds.shape[1]
    preds = torch.argmax(preds, dim=1)  # Convert to hard predictions
    targets = torch.argmax(targets, dim=1)  # Convert one-hot back to indices

    if class_weights is None:
        class_weights = torch.ones(num_classes, device=preds.device) / num_classes
    else:
        class_weights = class_weights / class_weights.sum()  # Normalize weights

    accuracy = torch.zeros(num_classes, device=preds.device)

    for c in range(num_classes):
        total = (targets == c).sum().float()
        if total > 0:
            correct = ((preds == c) & (targets == c)).sum().float()
            accuracy[c] = class_weights[c] * (correct / (total + eps))

    return accuracy.sum().item()


def get_weights(mask_dir: str, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for multi-class segmentation based on label frequency.

    Args:
        mask_dir (str): Path to the directory containing .npy segmentation masks.
        num_classes (int): Total number of classes in the dataset.

    Returns:
        torch.Tensor: A tensor containing computed class weights for each class.
    """
    train_mask_list = sorted(glob(f'{mask_dir}/*.npy'))
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for img_path in tqdm(train_mask_list, desc="Calculating class weights...", colour="green", leave=False):
        temp_image = np.load(img_path)  # Already one-hot encoded
        class_counts += temp_image.sum(axis=(0, 1, 2))  # Sum over spatial dimensions

    total_labels = class_counts.sum()

    # Compute class weights: n_samples / (n_classes * n_samples_for_class)
    class_weights = np.where(class_counts > 0, total_labels / (num_classes * class_counts), 0)

    # Normalize weights using softmax-like scaling
    class_weights = class_weights / np.max(class_weights)

    return torch.tensor(class_weights, dtype=torch.float32)


class HistoryDict(TypedDict):
    train_loss: List[float]
    train_dice: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_dice: List[float]
    val_acc: List[float]
    lr: List[float]


def plot_metrics(history: HistoryDict, save_plot: Optional[bool] = False) -> None:
    """
    Plot training metrics including loss, dice score, accuracy, and learning rate.

    Args:
        history (HistoryDict): Dictionary containing training history
        save_plot (Optional[bool]): If True, save a plot of the training history.
    """
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot Loss (Train vs Validation)
    axes[0, 0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0, 0].plot(history["val_loss"], label="Val Loss", color="red")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Over Epochs")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot Dice Score (Train vs Validation)
    axes[0, 1].plot(history["train_dice"], label="Train Dice Score", color="blue")
    axes[0, 1].plot(history["val_dice"], label="Val Dice Score", color="red")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice Score")
    axes[0, 1].set_title("Dice Score Over Epochs")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot Accuracy (Train vs Validation)
    axes[1, 0].plot(history["train_acc"], label="Train Accuracy", color="blue")
    axes[1, 0].plot(history["val_acc"], label="Val Accuracy", color="red")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy Over Epochs")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot Learning Rate
    axes[1, 1].plot(history["lr"], label="Learning Rate", color="green")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Over Epochs")
    axes[1, 1].set_yscale('log')  # Use log scale for learning rate
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_plot:
        plt.savefig('results/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
