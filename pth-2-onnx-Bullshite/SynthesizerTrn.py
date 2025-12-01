import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class Decoder(nn.Module):
    def __init__(self, hidden_channels, upsample_rates, resblock_channels, n_resblocks):
        super().__init__()
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        in_ch = hidden_channels
        for rate in upsample_rates:
            self.ups.append(nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=rate*2, stride=rate, padding=rate//2))
            in_ch = in_ch // 2
            self.resblocks.append(nn.ModuleList([ResBlock(resblock_channels) for _ in range(n_resblocks)]))
        self.post = nn.Conv1d(in_ch, 1, 7, padding=3)

    def forward(self, x):
        for up, resblock_group in zip(self.ups, self.resblocks):
            x = up(x)
            for rb in resblock_group:
                x = rb(x)
        x = self.post(x)
        return x

class SynthesizerTrn(nn.Module):
    def __init__(self, spec_channels, hidden_channels, n_layers, n_heads, upsample_rates, resblock_channels, n_resblocks):
        super().__init__()
        self.pre = nn.Conv1d(spec_channels, hidden_channels, 7, padding=3)
        self.decoder = Decoder(hidden_channels, upsample_rates, resblock_channels, n_resblocks)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, feats, p_len, pitch, pitchf, sid):
        """
        Args:
            feats: [B, C, T]
            p_len: [B]
            pitch: [B, 20] (dummy if unused)
            pitchf: [B, T] (dummy if unused)
            sid: [B] speaker ID (dummy if unused)
        Returns:
            audio waveform: [B, 1, T]
        """
        x = self.act(self.pre(feats))
        audio = self.decoder(x)
        return audio
