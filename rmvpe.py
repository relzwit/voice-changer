# rmvpe.py
# RMVPE (Robust Model for Vocal Pitch Estimation) Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RMVPE:
    def __init__(self, model_path, is_half=False, device='cpu'):
        """
        Initialize RMVPE pitch detector

        Args:
            model_path: Path to rmvpe.pt model file
            is_half: Use FP16 (not recommended for CPU)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.is_half = is_half

        print(f"[RMVPE] Loading model from {model_path}...")
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()

        if is_half and device == 'cuda':
            self.model = self.model.half()

        print("[RMVPE] Model loaded successfully")

    def infer_from_audio(self, audio, thred=0.03, sample_rate=16000):
        """
        Extract F0 from audio using RMVPE

        Args:
            audio: numpy array of audio samples (16kHz recommended)
            thred: Voicing threshold (lower = more sensitive)
            sample_rate: Sample rate of input audio

        Returns:
            numpy array of F0 values in Hz
        """
        # Ensure audio is the right shape
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension

        audio = audio.to(self.device)

        if self.is_half:
            audio = audio.half()

        # Run inference
        with torch.no_grad():
            # RMVPE expects specific input format
            # Pad audio to ensure proper frame alignment
            mel = self._wav2mel(audio, sample_rate)

            # Get pitch predictions
            hidden = self.model(mel)

            # Convert logits to F0
            f0 = self._decode_pitch(hidden, thred)

        return f0.squeeze().cpu().numpy()

    def _wav2mel(self, audio, sr=16000):
        """Convert waveform to mel spectrogram"""
        # RMVPE uses specific mel parameters
        n_fft = 2048
        hop_length = 160  # 10ms at 16kHz
        win_length = 2048
        n_mels = 128

        # Create mel filterbank
        mel_basis = torch.from_numpy(
            self._mel_filter_bank(sr, n_fft, n_mels)
        ).float().to(audio.device)

        # Compute STFT
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(audio.device),
            return_complex=True
        )

        # Magnitude spectrogram
        spec = torch.abs(spec)

        # Apply mel filterbank
        mel = torch.matmul(mel_basis, spec)

        # Log scaling
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel

    def _mel_filter_bank(self, sr, n_fft, n_mels, fmin=0, fmax=None):
        """Create mel filterbank matrix"""
        if fmax is None:
            fmax = sr / 2

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel points
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bins
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        # Create filterbank
        fbank = np.zeros((n_mels, n_fft // 2 + 1))

        for i in range(n_mels):
            left = bins[i]
            center = bins[i + 1]
            right = bins[i + 2]

            # Rising slope
            for j in range(left, center):
                fbank[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                fbank[i, j] = (right - j) / (right - center)

        return fbank

    def _decode_pitch(self, hidden, threshold=0.03):
        """
        Decode pitch from model output

        Args:
            hidden: Model output logits
            threshold: Voicing threshold

        Returns:
            F0 tensor
        """
        # RMVPE outputs 360 bins covering 50-1100 Hz
        # Bin centers are logarithmically spaced
        cents = torch.arange(360).float().to(hidden.device) * 20  # 20 cents per bin

        # Convert cents to Hz (C0 = 50 Hz reference)
        freq = 50 * (2 ** (cents / 1200))

        # Apply softmax to get probability distribution
        prob = F.softmax(hidden, dim=-1)

        # Weighted average to get pitch
        f0 = torch.sum(prob * freq, dim=-1)

        # Apply voicing threshold
        confidence = torch.max(prob, dim=-1)[0]
        f0 = torch.where(confidence > threshold, f0, torch.zeros_like(f0))

        return f0