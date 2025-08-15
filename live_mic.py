
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import librosa
from pathlib import Path
import soundfile as sf
import argparse

from models import MFCC1DCNN, MFCCLogisticRegression

def record_audio(duration=1.0, target_sr=8000):
    """
    Record audio from mic and return a dict similar to Hugging Face Audio feature:
    {
        'array': np.ndarray (float32, mono),
        'sampling_rate': int
    }
    """
    # Record from microphone
    y = sd.rec(int(duration * target_sr), samplerate=target_sr, channels=1, dtype='float32')
    sd.wait()
    
    # Flatten to 1D (mono)
    y = y.flatten()
    
    # Convert to float64 to match dataset dtype (optional, depends on training)
    y = y.astype(np.float64)

    # Return in HF Audio-like format

    return y

    
def to_mfcc(y, sr, n_mfcc = 64):
    # Compute MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )
    return mfcc 

n_mfcc = 64
model = MFCC1DCNN(num_classes=10, n_mfcc=n_mfcc)

def save_audio(y,sr, filename="recording.wav"):
    """
    audio_dict: {'array': np.ndarray, 'sampling_rate': int}
    filename: output file path
    """
    # Save as WAV
    sf.write(filename, y, sr)
    print(f"Saved recording to {filename}")
# model = MFCCLogisticRegression(num_classes=10, n_mfcc=n_mfcc)

checkpoint = torch.load('lightning_logs/CNN_default_mfcc/checkpoints/epoch=5-step=2028.ckpt', map_location='cpu')
# checkpoint = torch.load('lightning_logs/version_1/checkpoints/epoch=19-step=6760.ckpt', map_location='cpu')

device = torch.device('cpu')

if 'state_dict' in checkpoint:
    # Lightning checkpoint
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

duration = 0.5
print(f"Will record for a duration of {duration} secs. Press enter key to start recording.")
input()

audio = record_audio(duration=duration)
save_audio(audio, 8000)

mfcc = to_mfcc(audio, sr = 8000, n_mfcc=64)
mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(mfcc_tensor)
    pred = torch.argmax(logits, dim=1).item()


print(f"Predicted digit: {pred}")