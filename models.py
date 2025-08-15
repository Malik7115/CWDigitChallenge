
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_proc import SpokenDigitDataset

class MFCCLogisticRegression(nn.Module):
    def __init__(self, num_classes=10, n_mfcc=None):
        assert n_mfcc is not None, 'specify n_mfcc'
        super().__init__()
        self.classifier = nn.Linear(n_mfcc, num_classes)

    def forward(self, x):
        # x: (batch, n_mfcc, time)
        x = x.mean(dim=2)  # pool over time
        logits = self.classifier(x)
        return logits

class MFCC1DCNN(nn.Module):
    def __init__(self, num_classes=10, n_mfcc=None, hidden_channels=32, dropout=0.3):
        assert n_mfcc is not None, 'specify n_mfcc'
        super().__init__()
        
        # 1D CNN over time axis (input: n_mfcc features)
        self.conv1 = nn.Conv1d(in_channels=n_mfcc, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # x shape: (batch, n_mfcc, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)  # pool over time
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
    
class MelCNN(nn.Module):
    def __init__(self, n_mels=40, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # global pooling
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch, n_mels, time)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        logits = self.classifier(x)
        return logits

if __name__ == '__main__':

    dataset = SpokenDigitDataset(split="test", extract_features='mfcc', n_mfcc= 64)

    mfcc, label = dataset[0]
    mfcc = torch.tensor(mfcc[None, :])

    model = MFCCLogisticRegression()

    out = model(mfcc.float())
    x = 0


