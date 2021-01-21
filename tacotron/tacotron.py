import torch
import torch.nn as nn
from decoder import TCTRN_Decoder, TCTRN_Prenet, TCTRN_Postnet
from encoder import TCTRN_Encoder
from text.symbols import symbols
from math import sqrt
from utils.utils import to_gpu, get_mask_from_lengths


class TCTRN_Tacotron(nn.Module):
    def __init__(self , cfg):
        super(TCTRN_Tacotron, self).__init__()

        other_param = cfg["TCTRN"]["OTHER"]
        self.mask_padding = other_param["MASK_PADDING"]
        decoder_param = cfg["TCTRN"]["DECODER"]
        self.n_frams_per_step = decoder_param["N_FRAMES_PER_STEP"]
        audio_param = cfg["AUDIO"]
        self.n_mel_channels = audio_param["N_MEL_CHANNELS"]

        n_symbols = len(symbols) if other_param["N_SYMBOLS"] == -1 else other_param["N_SYMBOLS"]
        self.embedding = nn.Embedding(
            n_symbols, other_param["SYMBOLS_EMBD_DIM"]
        )
        # TODO
        std = sqrt(2.0 / (n_symbols + other_param["SYMBOLS_EMBD_DIM"]))
        val = sqrt(3.0) * std

        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = TCTRN_Encoder(cfg)
        self.decoder = TCTRN_Decoder(cfg)
        self.postnet = TCTRN_Postnet(cfg)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded)
        )

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            # TODO expand?
            mask = mask.expand(self.n_mel_channels, mask.size()[0], mask.size()[1])
            mask = mask.permute(1, 0, 2)

            outputs[0].data.mask_fill_(mask, 0.0)
            outputs[1].data.mask_fill_(mask, 0.0)
            outputs[2].data.mask_fill_(mask[:, 0, :], 1e3)

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        # TODO transpose
        embedding_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.embedding(embedding_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, text_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths
        )

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_output, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_output, alignments]
        )
        return outputs
