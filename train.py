import json
import os
from os import PathLike
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore, Style
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from dataset import BraTS2020Dataset as Dataset
from simple_3d_unet import Simple3DUNET
from utils import save_checkpoint, calc_dice_score, get_weights, calc_accuracy, load_checkpoint, plot_metrics

# ------------------------------
# Hyperparameters etc.
# ------------------------------

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
NUM_EPOCHS = 100
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True
# LOAD_MODEL = False
CLASS_WEIGHTS = get_weights("data/input_data_total/masks", num_classes=4)
TRAIN_IMG_DIR = "data/input_data_split/train/images"
TRAIN_MASK_DIR = "data/input_data_split/train/masks"
VAL_IMG_DIR = "data/input_data_split/val/images"
VAL_MASK_DIR = "data/input_data_split/val/masks"


# ------------------------------
# Utility Classes and Functions
# ------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation with pre-encoded masks
    """

    def __init__(self, weight=None, smooth=1e-5):
        """
        Args:
            weight (torch.Tensor): Class weights for loss calculation
            smooth (float): Small constant to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W, D)
        # target: (B, C, H, W, D) - already one-hot encoded

        # Convert predictions to probabilities
        pred = F.softmax(pred, dim=1)

        # Calculate Dice score for each class
        numerator = 2 * torch.sum(pred * target, dim=(2, 3, 4)) + self.smooth
        denominator = torch.sum(pred + target, dim=(2, 3, 4)) + self.smooth
        dice_score = numerator / denominator

        # Apply class weights if provided
        if self.weight is not None:
            dice_score = dice_score * self.weight.to(dice_score.device)

        # Average over classes and batches
        dice_loss = 1 - dice_score.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation with pre-encoded masks
    """

    def __init__(self, weight=None, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, pred, target):
        # pred: (B, C, H, W, D)
        # target: (B, C, H, W, D) - already one-hot encoded

        # Convert predictions to log probabilities
        log_prob = F.log_softmax(pred, dim=1)
        prob = torch.exp(log_prob)

        # Calculate the focal term
        focal = -self.alpha * torch.pow(1 - prob, self.gamma) * target * log_prob

        # Apply class weights if provided
        if self.weight is not None:
            focal = focal * self.weight.view(1, -1, 1, 1, 1).to(focal.device)

        # Average over all dimensions
        focal_loss = torch.mean(torch.sum(focal, dim=1))

        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Dice and Focal Loss for multi-class segmentation
    """

    def __init__(
            self,
            weight=None,
            dice_weight=0.5,
            focal_weight=0.5,
            dice_smooth=1e-5,
            focal_gamma=2.0,
            focal_alpha=0.25
    ):
        """
        Args:
            weight (torch.Tensor): Class weights for loss calculation
            dice_weight (float): Weight for Dice Loss
            focal_weight (float): Weight for Focal Loss
            dice_smooth (float): Smoothing factor for Dice Loss
            focal_gamma (float): Focusing parameter for Focal Loss
            focal_alpha (float): Balancing parameter for Focal Loss
        """

        super(CombinedLoss, self).__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(weight=weight, smooth=dice_smooth)
        self.focal_loss = FocalLoss(weight=weight, gamma=focal_gamma, alpha=focal_alpha)

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Model predictions of shape (B, C, H, W)
            target (torch.Tensor): Ground truth of shape (B, H, W)

        Returns:
            torch.Tensor: Combined loss value
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        return self.dice_weight * dice + self.focal_weight * focal


class EarlyStopping:
    """
    Early stops the training if the validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # If validation loss is not a number, skip
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        save_checkpoint(model.state_dict(), self.path, verbose=False)
        self.val_loss_min = val_loss


def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        class_weights: torch.Tensor = CLASS_WEIGHTS,
        device: str | torch.device = DEVICE
) -> Tuple[float, Tuple[float, float]]:
    """
    Perform a single training step for a model on a multi-class segmentation task using weighted metrics.

    Args:
        model (nn.Module): The 3D U-Net model.
        dataloader (DataLoader): The DataLoader for training data.
        optimizer (optim.Optimizer): The optimizer used to update the model parameters.
        loss_fn (nn.Module): The loss function.
        class_weights (torch.Tensor): Weights for each class.
        device (str | torch.device): The device to run the model on, 'cuda' or 'cpu'.

    Returns:
        Tuple[float, Tuple[float, float]]: The average training loss and a tuple of weighted Dice score and accuracy.
    """
    model.train()
    train_loss, total_dice, total_acc = 0.0, 0.0, 0.0

    for batch, (X, y) in enumerate(pbar := tqdm(dataloader, desc='Training', colour="green", leave=False)):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # Metrics computation
        train_loss += loss.item()
        total_dice += calc_dice_score(y_pred, y, class_weights)
        total_acc += calc_accuracy(y_pred, y, class_weights)

        # Update pbar
        pbar.set_postfix(loss=train_loss / (batch + 1), dice=total_dice / (batch + 1), acc=total_acc / (batch + 1))

    avg_loss = train_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    return avg_loss, (avg_dice, avg_acc)


def test_step(
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        class_weights: torch.Tensor = CLASS_WEIGHTS,
        device: str | torch.device = DEVICE
) -> Tuple[float, Tuple[float, float]]:
    """
    Perform a single validation step for the model.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The DataLoader for testing data.
        loss_fn (nn.Module): The loss function, default is CrossEntropyLoss.
        class_weights (torch.Tensor): Weights for each class.
        device (str): The device to run the model on, 'cuda' or 'cpu'.

    Returns:
        Tuple[float, Tuple[float, float]]: The average training loss and a tuple of weighted Dice score and accuracy.
    """
    model.eval()
    test_loss, total_dice, total_acc = 0.0, 0.0, 0.0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(pbar := tqdm(dataloader, desc='Validating', colour="green", leave=False)):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # Metrics computation
            test_loss += loss.item()
            total_dice += calc_dice_score(y_pred, y, class_weights)
            total_acc += calc_accuracy(y_pred, y, class_weights)

            # Update pbar
            pbar.set_postfix(loss=test_loss / (batch + 1), dice=total_dice / (batch + 1), acc=total_acc / (batch + 1))

    avg_loss = test_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    return avg_loss, (avg_dice, avg_acc)


def train_fn(
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: nn.Module,
        device: torch.device = DEVICE,
        epochs: int = 10,
        best_checkpoint: str | PathLike[str] = "checkpoints/checkpoint.pth",
        patience: int = 5
):
    """
    Train and evaluate the model over multiple epochs and applies early stopping based on validation loss.

    Args:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): The DataLoader for training data.
        val_dataloader (DataLoader): The DataLoader for testing/validation data.
        optimizer (Optimizer): The optimizer used to update the model parameters.
        scheduler (lr_scheduler): The learning rate scheduler to adjust the learning rate.
        loss_fn (nn.Module): The loss function.
        device (torch.device): The device to run the model on.
        epochs (int): The number of epochs to train the model.
        best_checkpoint (str): Path to save the model checkpoints.
        patience (int): Number of epochs to wait before early stopping.

    Returns:
        dict: A dictionary containing training and validation metrics for each epoch.
    """
    early_stopping = EarlyStopping(patience=patience, path=best_checkpoint, trace_func=tqdm.write, verbose=True)

    history = {
        "train_loss": [],
        "train_dice": [],
        "train_acc": [],
        "val_loss": [],
        "val_dice": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(epochs):
        tqdm.write(f"\n======== Epoch {epoch + 1}/{epochs} ========")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        # Training phase
        train_loss, (train_dice, train_acc) = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            class_weights=CLASS_WEIGHTS,
            device=device
        )

        # Validation phase
        val_loss, (val_dice, val_acc) = test_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            class_weights=CLASS_WEIGHTS,
            device=device
        )

        # Update history
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc)

        # Save model checkpoint after every epoch
        save_checkpoint(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pth", trace_fn=tqdm.write)

        # Print epoch results
        tqdm.write(
            f"Train Loss: {train_loss:.6f}, Train Dice: {train_dice:.6f}, Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f}, Val Dice: {val_dice:.6f}, Val Acc: {val_acc:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Step the learning rate scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            tqdm.write(f"{Fore.YELLOW}Early stopping triggered!{Style.RESET_ALL}")
            break

    # Load the best model weights from checkpoint
    load_checkpoint(early_stopping.path, model, trace_fn=tqdm.write)
    tqdm.write(f"{Fore.GREEN}Training complete!{Style.RESET_ALL}")
    return history


# ------------------------------
# Main Function
# ------------------------------

def main():
    print(f"Running on device: {Fore.GREEN}'{DEVICE}'{Style.RESET_ALL}.")
    print(f"Class Weights: {CLASS_WEIGHTS}")

    # Load Dataset
    train_dataset = Dataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR)
    val_dataset = Dataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR)

    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                shuffle=False)

    # Create model
    model = Simple3DUNET(in_channels=3, out_channels=4).to(DEVICE)
    summary(model, input_size=(4, 3, 128, 128, 128))

    # Loss fn, optimizer, etc.
    loss_fn = CombinedLoss(
        weight=CLASS_WEIGHTS,
        dice_weight=0.5,  # Equal weighting between Dice and Focal loss
        focal_weight=0.5,
        dice_smooth=1e-5,
        focal_gamma=2.0,  # Standard focal loss gamma
        focal_alpha=0.25  # Standard focal loss alpha
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Number of iterations for the first restart
        T_mult=2,  # A factor increases the number of iterations after a restart
        eta_min=1e-6  # Minimum learning rate
    )

    # Start training model
    history = train_fn(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        best_checkpoint=f"checkpoints/best_model_{NUM_EPOCHS}.pth"
    )

    # Plot and save results
    plot_metrics(history)
    with open("results/history.json", "w") as f:
        # noinspection PyTypeChecker
        json.dump(history, f)


if __name__ == "__main__":
    main()
