import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse

from models import MFCC1DCNN, MFCCLogisticRegression
from data_proc import SpokenDigitDataset
from time import time
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score


import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Testing")

    parser.add_argument("--n_features", type=str, required=False, default=15,
                        help="num features for MFCC, default is 15")
    
    parser.add_argument("--ckpt_path", type=str, required=False, default='logs/best-cnn-mfcc/checkpoints/last.ckpt',
                    help="model ckpt path for inference, defaults to best CNN model")
    
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    n_features = args.n_features

    test_dataset = SpokenDigitDataset(split="test", extract_features="mfcc", n_mfcc= n_features)
    
    device = torch.device('cpu')
    model = MFCC1DCNN(n_mfcc=n_features)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    


    if 'state_dict' in checkpoint:
    # Lightning checkpoint
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    times = []
    preds = []
    labels = []

    for i, sample in enumerate(test_dataset):
        t1 = time()
        mfcc, label = sample
        mfcc = torch.tensor(mfcc[None,:]).float()
        with torch.no_grad():
            logits = model(mfcc)
            pred = torch.argmax(logits, dim=1).item()
            # print(f'label: {label}, pred: {pred}')
        preds.append(pred)
        labels.append(label)

        t2 = time()
        times.append(t2-t1)

    accuracy  = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
    print(f'max sample time {max(times)}, min sample tim {min(times)}, Average Time:{sum(times)/len(times)}, Total time:{sum(times)}')



