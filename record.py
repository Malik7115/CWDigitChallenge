import pyaudio
import webrtcvad
import numpy as np
from scipy.signal import resample_poly
import wave

class VADRecorder:
    def __init__(self, aggressiveness=2, rate=16000, frame_duration=30, silence_limit=10):
        """
        aggressiveness: VAD aggressiveness level (0–3)
        rate: input sample rate (must be 16000 for webrtcvad)
        frame_duration: frame size in ms (10, 20, or 30)
        silence_limit: number of silent frames before stopping
        """
        self.rate = rate
        self.frame_duration = frame_duration
        self.frame_size = int(rate * frame_duration / 1000)
        self.channels = 1
        self.silence_limit = silence_limit

        self.vad = webrtcvad.Vad(aggressiveness)
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16,
                                   channels=self.channels,
                                   rate=self.rate,
                                   input=True,
                                   frames_per_buffer=self.frame_size)

        print("🎤 VADRecorder initialized — start speaking!")

    def record(self):
        """Record until silence is detected, return trimmed NumPy array (8kHz mono)."""
        frames = []
        silence_counter = 0
        started = False

        while True:
            audio = self.stream.read(self.frame_size, exception_on_overflow=False)
            is_speech = self.vad.is_speech(audio, self.rate)

            if is_speech:
                frames.append(audio)
                started = True
                silence_counter = 0
            elif started:
                silence_counter += 1
                if silence_counter > self.silence_limit:
                    break

        # Combine to raw bytes → NumPy int16
        audio_bytes = b"".join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # Trim leading & trailing silence
        audio_trimmed = self._trim_silence(audio_np)

        # Resample to 8kHz mono
        audio_8k = self._resample_to_8k(audio_trimmed)

        return audio_8k

    def _trim_silence(self, audio_np, threshold=500):
        """Remove leading/trailing silence based on amplitude threshold"""
        abs_audio = np.abs(audio_np)
        non_silent = np.where(abs_audio > threshold)[0]
        if len(non_silent) == 0:
            return audio_np
        start, end = non_silent[0], non_silent[-1] + 1
        return audio_np[start:end]

    def _resample_to_8k(self, audio_np):
        """Downsample to 8kHz mono int16"""
        gcd = np.gcd(self.rate, 8000)
        up = 8000 // gcd
        down = self.rate // gcd
        resampled = resample_poly(audio_np, up, down)
        return resampled.astype(np.int16)

    def save(self, filename, audio_np, rate=8000):
        """Save NumPy audio array to WAV"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(audio_np.tobytes())
        print(f"💾 Saved {filename} ({rate} Hz)")

    def close(self):
        """Clean up stream and PyAudio"""
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        print("🛑 Stream closed")
