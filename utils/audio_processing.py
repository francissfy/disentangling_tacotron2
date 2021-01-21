import torch
import numpy as np
import librosa.util as librosa_util
from scipy.signal import get_window


# TODO
def window_sumsquare(window,
                     n_frames,
                     hop_length=120,
                     win_length=800,
                     n_fft=800,
                     dtype=float,
                     norm=None):
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample: min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    return torch.exp(x) / C
