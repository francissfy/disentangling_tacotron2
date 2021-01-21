import torch
import random
import numpy as np
from torch.autograd import Variable
from typing import List
from tacotron.layers import TCTRN_Stft
from text import text_to_sequence
from utils.utils import load_wav_to_torch

"""
这都不是问题
前处理一下就好了
"""
"""
def get_VCTK_audio_text(line: str, base_dir: str) -> Tuple[str, str]:
    tmp = line.split(sep=" ", maxsplit=1)
    audio_id, text = tmp[0], tmp[1]
    spk_id = audio_id.split(sep="_", maxsplit=1)[0]
    audio_path = os.path.join(base_dir, "wav", spk_id, f"{audio_id}.wav")
    return audio_path, text


def get_LJ_audio_text(line: str, base_dir: str) -> Tuple[str, str]:
    tmp = line.split(sep="|", maxsplit=1)
    audio_id, text = tmp[0], tmp[1]
    audio_path = os.path.join(base_dir, "wavs", f"{audio_id}.wav")
    return audio_path, text
"""


def load_audiopath_text(file: str):
    return [["", ""]]


# JetBrain SB 明明有还报错
class TextMelLoader(torch.utils.data.Dataset):
    """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.

    cfg: see param/config.yaml
    """
    def __init__(self, audio_text_file: str, cfg):
        # TODO type of audiopath_text
        self.audiopath_text: List[List[str]] = load_audiopath_text(audio_text_file)
        self.text_cleaners: List[str] = cfg["UTILS"]["UTILS_TEXT_CLEANERS"]
        self.max_wav_value: float = cfg["AUDIO"]["MAX_WAV_VALUE"]
        self.sampling_rate: float = cfg["AUDIO"]["SAMPLING_RATE"]
        self.load_mel_from_disk = cfg["DATA"]["LOAD_MEL_FROM_DISK"]
        self.stft = TCTRN_Stft(
            filter_length=cfg[""][""],
            hop_length=cfg["AUDIO"]["HOP_LENGTH"],
            win_length=cfg["AUDIO"]["WIN_LENGTH"],
            n_mel_channels=cfg["AUDIO"]["N_MEL_CHANNELS"],
            sampling_rate=cfg["AUDIO"]["SAMPLING_RATE"],
            mel_fmin=cfg["AUDIO"]["MEL_FMIN"],
            mel_fmax=cfg["AUDIO"]["MEL_FMAX"]
        )
        random.seed(cfg["EXPERIMENT"]["SEED"])
        random.shuffle(self.audiopath_text)

    def get_mel_text_pair(self, audiopath_and_text: List[str]):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        mel = self.get_mel(audiopath)
        text = self.get_text(text)
        return mel, text

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(f"dataloader.py: get_mel: "
                                 f"{sampling_rate} vs {self.stft.sampling_rate} sampling rate doesn't match")
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec - torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size()[0] == self.stft.n_mel_channels, (
                f"dataloader.py: get_mel: mel dimension mismatch: {melspec.size(0)} vs {self.stft.n_mel_channels}"
            )
        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopath_text[index])

    def __len__(self):
        return len(self.audiopath_text)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [[text_normalized, mel_normalized], ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size()[0]] = text

        # Right pad melspec with 0
        num_mels = batch[0][1].size()[0]
        max_target_len = max([x[1].size()[1] for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gated_padded = torch.FloatTensor(len(batch), max_target_len)
        gated_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size()[1]] = mel
            gated_padded[i, mel.size()[1]-1:] = 1
            output_lengths[i] = mel.size()[1]
        return text_padded, input_lengths, mel_padded, gated_padded, output_lengths




