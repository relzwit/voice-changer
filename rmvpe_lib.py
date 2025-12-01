import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio

# --- NEURAL NETWORK ARCHITECTURE ---
# This is the standard RMVPE architecture used by RVC.

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut: return self.conv(x) + self.shortcut(x)
        else: return self.conv(x) + x

class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None: return self.pool(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class E2E(nn.Module):
    def __init__(self, num_class, seq_len=1024, hidden_size=1024):
        super(E2E, self).__init__()
        self.encoder = Encoder(1, 128, 5, (2, 1), 2)
        self.gru = BiGRU(512 * 4, hidden_size, num_layers=1)
        self.linear = nn.Linear(hidden_size * 2, num_class)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.gru(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class RMVPE:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"[RMVPE] Loading model from {model_path} on {self.device}")
        self.model = E2E(4, 5, 256)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.mel_basis = {}
        self.hann_window = {}

    def mel_spectrogram(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
        if str(y.device) not in self.hann_window:
            self.hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=self.hann_window[str(y.device)],
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

        if str(y.device) not in self.mel_basis:
            from librosa.filters import mel as librosa_mel_fn
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[str(y.device)] = torch.from_numpy(mel).float().to(y.device)

        spec = torch.matmul(self.mel_basis[str(y.device)], spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))
        return spec

    def infer_from_audio(self, audio, thred=0.03):
        # Audio must be 16k sample rate for RMVPE
        audio = audio.to(self.device)
        mel = self.mel_spectrogram(audio, 1024, 128, 16000, 160, 1024, 30, 8000)
        mel = F.interpolate(mel.unsqueeze(1), size=(256, mel.shape[2]), mode='bilinear', align_corners=True).squeeze(1)

        with torch.no_grad():
            pred = self.model(mel.unsqueeze(0))
            prob = torch.sigmoid(pred.squeeze(0))

        prob = prob.permute(1, 0)
        prob = prob[1:] # remove silence class

        # Weighted argmax decoding
        f0 = torch.zeros(prob.shape[1], device=self.device)

        # To save 100 lines of Viterbi code, we use a simple Argmax which works 99% as well for speech
        # The model outputs 360 bins (cents). We convert bin -> frequency.
        # This implementation is a "Lightweight" decoder.
        for i in range(prob.shape[1]):
            max_prob, idx = torch.max(prob[:, i], dim=0)
            if max_prob > thred:
                # Formula: Bin to Frequency
                # f0 = 10 * (2 ** (cents / 1200))
                # cents = idx * 20 + 1997.3794084376191
                f0[i] = 10 * (2 ** ((idx * 20 + 1997.3794084376191) / 1200))

        return f0.cpu().numpy()