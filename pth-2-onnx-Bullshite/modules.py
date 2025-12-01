import torch
import torch.nn as nn
import torch.nn.functional as F


# RVC V2 Mish activation
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels, channels, k,
                        dilation=d, padding=get_padding(k, d)
                    ),
                    nn.LeakyReLU(0.1),
                )
            )

    def forward(self, x):
        for conv in self.convs:
            xt = conv(x)
            x = xt + x
        return x
