# CREATED BY SHEN FEIYU
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TCTRN_Conv1d, TCTRN_Linear


class TCTRN_Encoder(nn.Module):
    def __init__(self, cfg):
        super(TCTRN_Encoder, self).__init__()
        # cfg for encoder
        encoder_param = cfg["TCTRN"]["ENCODER"]
        # conv set
        conv_set = []
        for _ in range(encoder_param["NUM_CONV"]):
            conv = TCTRN_Conv1d(encoder_param["EMBD_DIM"],
                                encoder_param["EMBD_DIM"],
                                encoder_param["KERNEL_SIZE"],
                                1,
                                int((encoder_param["KERNEL_SIZE"]-1)/2),
                                1,
                                init_gain="relu"
                                )
            bn = nn.BatchNorm1d(encoder_param["EMBD_DIM"])
            conv_set.append(nn.Sequential(
                conv, bn
            ))
        # conv group
        self.convolutions = nn.ModuleList(conv_set)
        # lstm, output shape: same as input
        self.lstm = nn.LSTM(input_size=encoder_param["EMBD_DIM"],
                            hidden_size=int(encoder_param["EMBD_DIM"]/2),
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        
    def forward(self, x: torch.FloatTensor, input_lengths: torch.IntTensor):
        # conv -> relu -> dropout
        for conv in self.convolutions:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
        # TODO 这里在干嘛
        x = x.transpose(1, 2)
        
        input_lengths = input_lengths.cpu().numpy()
        # pack the padded sequence for lstm
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)
        # lstm
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        # recover from packed sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True)
        return out

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        # 这里没有pack跟pad 估计是为了加快inference速度 不过为了效果还是加上比较好吧
        out, _ = self.lstm(x)
        return out
