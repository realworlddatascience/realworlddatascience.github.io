import os
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_early_stopping_callback(metric, patience):
    """Early Stopping Callback"""
    return EarlyStopping(
        monitor=metric if "val" in metric else f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True
    )


def get_checkpoint_callback(output_dir, filename, metric, save_top_k=1):
    output_dir = os.path.join(output_dir, "ann/")
    file_pt = os.path.join(output_dir, f"{filename}.ckpt")

    # Lightning is not overwriting it across multiple runs.
    # So manually remove the file if it exists
    if os.path.exists(file_pt):
        os.remove(file_pt)

    return ModelCheckpoint(
        dirpath=output_dir,
        filename=filename,
        monitor=metric if "val" in metric else f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        verbose=True
    )


class ANNLoggingCallback(pl.Callback):
    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f"\nFit Starts\n")

    @rank_zero_only
    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Validation Sanity Check Start\n")

    @rank_zero_only
    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Validation Sanity Check End\n")

    @rank_zero_only
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("\nTrain Epoch Start\n")

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("\nTrain Epoch Done\n")

    @rank_zero_only
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("\nTraining Starts\n")

    @rank_zero_only
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("\nTraining Done\n")

    @rank_zero_only
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("\nValidation Epoch Start\n")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("\nValidation Epoch Done\n")


