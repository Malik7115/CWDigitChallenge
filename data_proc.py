from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import io
import librosa
import torch
import random

class SpokenDigitDataset(Dataset):
    def __init__(self, split="train", sr=8000, n_mfcc=13, extract_features= None, augment= True):
        """
        split: 'train' or 'test'
        sr: target sampling rate
        n_mfcc: number of MFCC coefficients
        extract_features: if True, return MFCC features; else return raw audio
        """
        dataset = load_dataset("mteb/free-spoken-digit-dataset", split="train", cache_dir='./data/')

        # Create train/val split
        split_data = dataset.train_test_split(test_size=0.15, seed=42)

        if split == 'train':
            self.dataset = split_data["train"]
        elif split == 'val':
            self.dataset = split_data["test"]
        elif split == 'test':
            # If the dataset had a test split, you could load it separately
            self.dataset = load_dataset("mteb/free-spoken-digit-dataset", split="test", cache_dir='./data/')

        else:
            raise ValueError(f"Unknown split: {split}")


        self.sr = sr
        self.augment = augment
        self.n_mfcc = n_mfcc
        self.split  = split
        self.extract_features = extract_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get raw audio bytes
        # audio_bytes = self.dataset[idx]['audio']._hf_encoded['bytes']
        # Decode using soundfile
        y, orig_sr = self.dataset[idx]['audio']['array'], self.dataset[idx]['audio']['sampling_rate']
        
        # Ensure mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        # Resample
        if orig_sr != self.sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)

        
        if self.augment and self.split == "train":
            y = self.apply_augmentations(y)

        label = int(self.dataset[idx]["label"])

        if self.extract_features == "mfcc":
            features = self.to_mfcc(y)
            return features, label
        elif self.extract_features == "melspec":
            features = self.to_melspec(y)
            return features, label
        else:
            return y, label
        
    def to_melspec(self, y, n_mels=40):
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db  # keep as 2D array
    
    def to_mfcc(self, y):
        # Compute MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc
        )
        return mfcc 

    def apply_augmentations(self, y):
        # Random additive noise
        if random.random() < 0.75:
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            y = y + noise_amp * np.random.normal(size=y.shape)

        # Random pitch shift
        if random.random() < 0.30:
            n_steps = np.random.uniform(-2, 2)  # semitones
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

        # Random time stretch
        if random.random() < 0.30:
            rate = np.random.uniform(0.8, 1.25)
            y = librosa.effects.time_stretch(y, rate=rate)

        # Clip to valid range
        y = np.clip(y, -1.0, 1.0)
        return y


class AudioPadCollator:
    def __init__(self, pad_value=0.0):
        self.pad_value = pad_value

    def __call__(self, batch):
        """
        batch: list of (features, label) tuples
        features: np.ndarray (1D waveform or 2D feature matrix)
        """
        features, labels = zip(*batch)

        # Detect if we have 1D (waveform) or 2D (spectrogram) data
        if features[0].ndim == 1:  
            # 1D waveforms: pad along length
            max_len = max(f.shape[0] for f in features)
            padded = [np.pad(f, (0, max_len - len(f)), mode='constant', constant_values=self.pad_value)
                      for f in features]
            features_tensor = torch.tensor(np.stack(padded), dtype=torch.float32)

        elif features[0].ndim == 2:
            # 2D features: pad along time axis (axis=1 for librosa features)
            max_len = max(f.shape[1] for f in features)
            padded = [np.pad(f, ((0, 0), (0, max_len - f.shape[1])), mode='constant', constant_values=self.pad_value)
                      for f in features]
            features_tensor = torch.tensor(np.stack(padded), dtype=torch.float32)

        else:
            raise ValueError("Unsupported feature shape: {}".format(features[0].shape))

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return features_tensor, labels_tensor






if __name__ == "__main__":

    collator = AudioPadCollator(pad_value=0.0)
    loader = DataLoader(
    SpokenDigitDataset(split="test", extract_features=None),
    batch_size=2, # for testing
    shuffle=True,
    collate_fn=collator
    )

    # dataset = SpokenDigitDataset(extract_features="melspec")
    
    for batch in loader:
        mfcc_batch, labels_batch = batch
        x = 0


