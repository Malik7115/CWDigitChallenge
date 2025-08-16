import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_proc import SpokenDigitDataset, AudioPadCollator
from torch.utils.data import DataLoader
from models import MFCCLogisticRegression, MFCC1DCNN
from torchmetrics import ConfusionMatrix, Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")

pl.seed_everything(42, workers=True)

class LightningAudioClassifier(pl.LightningModule):
    def __init__(self, model, num_classes=10, lr=1e-3):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes

        # Torchmetrics for epoch-level accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Confusion matrix for test
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        self.log("train_acc", acc, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.log("val_acc", acc, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)

        self.test_acc.update(preds, y)
        self.test_confmat.update(preds, y)

    def on_test_epoch_end(self):
        # Log test accuracy
        acc = self.test_acc.compute()
        self.log("test_acc", acc, prog_bar=True)
        self.test_acc.reset()

        # Compute confusion matrix
        confmat = self.test_confmat.compute().cpu().numpy()
        self.test_confmat.reset()

        fig, ax = plt.subplots()
        im = ax.imshow(confmat, cmap="Blues")
        ax.set_title("Test Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.renderer.buffer_rgba())
        image = buf[:, :, :3] / 255.0

        self.logger.experiment.add_image(
            "ConfusionMatrix", image, global_step=self.current_epoch, dataformats="HWC"
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train MFCC CNN classifier on spoken digit dataset")

    # Features / dataset
    parser.add_argument("--n_features", type=int, default=64, help="Number of MFCC features")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Test batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")

    # Model
    parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden channels in CNN")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Training
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator: 'gpu' or 'cpu'")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (GPUs/CPUs)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()

    # Datasets
    train_dataset = SpokenDigitDataset(split="train", extract_features="mfcc", n_mfcc=args.n_features)
    val_dataset   = SpokenDigitDataset(split="val", extract_features="mfcc", n_mfcc=args.n_features)
    test_dataset  = SpokenDigitDataset(split="test", extract_features="mfcc", n_mfcc=args.n_features)

    # Collator
    collator = AudioPadCollator(pad_value=0.0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.val_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=args.test_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

    # Model
    model = MFCC1DCNN(n_mfcc=args.n_features, hidden_channels=args.hidden_channels, dropout=args.dropout)
    lit_module = LightningAudioClassifier(model, num_classes=10, lr=args.lr)

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="last",
        save_last=True,
        save_top_k=0
    )
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback]
    )
    # Training & Testing
    trainer.fit(lit_module, train_loader, val_loader)
    trainer.test(lit_module, test_loader)

if __name__ == '__main__':
    main()


