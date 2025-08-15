import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_proc import SpokenDigitDataset, AudioPadCollator
from torch.utils.data import DataLoader
from models import MFCCLogisticRegression, MFCC1DCNN
from torchmetrics import ConfusionMatrix
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class LightningAudioClassifier(pl.LightningModule):
    def __init__(self, model, num_classes=10, lr=1e-3):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes

        # Torchmetrics confusion matrix
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        self.test_confmat.update(preds, y)

    def on_test_epoch_end(self):
        confmat = self.test_confmat.compute().cpu().numpy()

        fig, ax = plt.subplots()
        im = ax.imshow(confmat, cmap="Blues")
        ax.set_title("Test Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)

        # Convert Matplotlib figure to numpy image
        fig.canvas.draw()

        # Get RGBA buffer as a numpy array
        buf = np.asarray(fig.canvas.renderer.buffer_rgba())

        # Convert RGBA to RGB by dropping the alpha channel
        image = buf[:, :, :3]

        # Normalize to [0,1] float for TensorBoard
        image = image / 255.0

        self.logger.experiment.add_image(
            "ConfusionMatrix", image, global_step=self.current_epoch, dataformats="HWC"
        )

        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
    

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)

    n_features= 64
    train_dataset = SpokenDigitDataset(split="train", extract_features="mfcc", n_mfcc=n_features)
    val_dataset = SpokenDigitDataset(split="test", extract_features="mfcc", n_mfcc= n_features)

    # Collator
    collator = AudioPadCollator(pad_value=0.0)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collator, num_workers= 4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collator, num_workers= 4)

    # model = MFCCLogisticRegression(n_mfcc=n_mfcc)
    model = MFCC1DCNN(n_mfcc=n_features, hidden_channels=8, dropout=0.25)

    lit_module = LightningAudioClassifier(model, num_classes=10, lr=1e-3)

    checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",       # folder to save in
    filename="last",             # name for checkpoint file (without extension)
    save_last=True,               # <--- saves only last.ckpt
    save_top_k=0                  # don't save "best" checkpoints
    )

    trainer = pl.Trainer(max_epochs= 20, accelerator="gpu", devices=1)

    trainer.fit(lit_module,
                train_loader,
                val_loader,
    )
    trainer.test(lit_module, val_loader)

