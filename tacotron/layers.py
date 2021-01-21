import torch
import torch.nn as nn
from stft import STFT
from utils.audio_processing import dynamic_range_compression, dynamic_range_decompression
from librosa.filters import mel as librosa_mel_fn


class TCTRN_Conv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = None,
                 dilation: int = 1,
                 bias: bool = False,
                 init_gain: str = "linear"):
        super(TCTRN_Conv1d, self).__init__()
        if padding is not None:
            # 大概是为了整除吧
            assert (kernel_size % 2 == 1)
            # keep the output shape the same as the input
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        # init weight
        nn.init.xavier_uniform_(self.conv.weight,
                                gain=nn.init.calculate_gain(init_gain))

    def forward(self, x: torch.FloatTensor):
        out = self.conv(x)
        return out


class TCTRN_Linear(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 bias: bool = True,
                 init_gain: str = "linear"):
        super(TCTRN_Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        nn.init.xavier_uniform_(self.linear.weight,
                                gain=nn.init.calculate_gain(init_gain))

    def forward(self, x: torch.FloatTensor):
        out = self.linear(x)
        return out


class TCTRN_Stft(torch.nn.Module):
    def __init__(self,
                 filter_length: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mel_channels: int = 80,
                 sampling_rate: int = 22050,
                 mel_fmin: float = 0.0,
                 mel_fmax: float = 8000.0):
        super(TCTRN_Stft, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)

        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) < 1)

        magnitudes, phase = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
