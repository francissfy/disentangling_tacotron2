import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import TCTRN_Linear, TCTRN_Conv1d
from utils.utils import get_mask_from_lengths


class TCTRN_Location(nn.Module):
    """
    extract feature from previous attentions through conv -> linear
    location_inner_dim: output dim of conv layer
    """
    def __init__(self,
                 location_inner_dim: int,
                 location_kernel_size: int,
                 location_out_dim: int):
        super(TCTRN_Location, self).__init__()
        padding: int = int((location_kernel_size - 1) / 2)
        self.conv = TCTRN_Conv1d(in_channels=2,
                                 out_channels=location_inner_dim,
                                 kernel_size=location_kernel_size,
                                 padding=padding,
                                 bias=False,
                                 stride=1,
                                 dilation=1)
        self.linear = TCTRN_Linear(in_dim=location_inner_dim,
                                   out_dim=location_out_dim,
                                   bias=False,
                                   init_gain="tanh")

    def forward(self, attention_weights_cat):
        # (B, 2, max_time) -> (B, location_inner_dim, max_time)
        processed_attention = self.conv(attention_weights_cat)
        # (B, location_inner_dim, max_time) -> (B, max_time, location_inner_dim)
        processed_attention = processed_attention.transpose(1, 2)
        # (B, max_time, location_inner_dim) -> (B, max_time, location_out_dim)
        processed_attention = self.linear(processed_attention)
        return processed_attention


class TCTRN_Attention(nn.Module):
    def __init__(self,
                 attention_rnn_dim: int,
                 embedding_dim: int,
                 attention_dim: int,
                 location_inner_dim: int,
                 location_kernel_size: int):
        super(TCTRN_Attention, self).__init__()
        self.query_layer = TCTRN_Linear(in_dim=attention_rnn_dim,
                                        out_dim=attention_dim,
                                        bias=False,
                                        init_gain="tanh")
        self.memory_layer = TCTRN_Linear(in_dim=embedding_dim,
                                         out_dim=attention_dim,
                                         bias=False,
                                         init_gain="tanh")
        self.linear_layer = TCTRN_Linear(in_dim=attention_dim,
                                         out_dim=1,
                                         bias=False)
        self.location_layer = TCTRN_Location(location_inner_dim=location_inner_dim,
                                             location_kernel_size=location_kernel_size,
                                             location_out_dim=attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment(self, query, processed_memory, attention_weights_cat):
        """get the attention weight
        query: decoder output, shape (B, N_Mel_Channel * N_Frame_Per_Step), N_Mel_Channel * N_Frame_Per_Step
            aka attention_dim
        processed_memory: processed encoder output, shape (B, T_in, attention_dim), T_in aka 1
        attention_weights_cat: cumulative and prev, shape (B, 2, max_time)
        """
        # state of the generator
        # (B, NMC*NFPS) -> (B, 1, NMC*NFPS) -> (B, 1, attention_dim)
        processed_query = self.query_layer(query.unsqueeze(1))
        # location awareness
        # (B, 2, max_time) -> (B, max_time, attention_dim)
        processed_attention_weights = self.location_layer(attention_weights_cat)
        # from <Attention-Based Models for Speech Recognition>
        alignment = self.linear_layer(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))  # alignment shape (B, max_time, 1)
        alignment = alignment.squeeze(-1)
        # (B, max_time)
        return alignment

    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask):
        """
        memory: encoder otuput, shape (B, max_time, embedding_dim)
        attention_hidden_state: attention rnn last output, aka query
        mask: binary mask for padded data
        """
        # (B, max_time), [0...max_time] encoder output
        alignment = self.get_alignment(
            attention_hidden_state, processed_memory, attention_weights_cat)
        if mask is not None:
            # mark the padded area
            alignment.data.mask_fill_(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)
        # (B, 1, max_time) * (B, max_time, embedding_dim) -> (B, 1, embedding_dim)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights


class TCTRN_Prenet(nn.Module):
    def __init__(self, in_dim: int, out_dims):
        super(TCTRN_Prenet, self).__init__()
        in_dims = [in_dim] + out_dims[:-1]
        # TODO ModuleList here vs Sequential?
        self.linear_layers = nn.ModuleList([
            TCTRN_Linear(in_dim=t1, out_dim=t2, bias=False)
            for (t1, t2) in zip(in_dims, out_dims)
        ])

    def forward(self, x):
        for layer in self.linear_layers:
            x = F.relu(layer(x))
            # TODO training=True even in inference?
            x = F.dropout(x, p=0.5, training=True)
        return x


class TCTRN_Postnet(nn.Module):
    def __init__(self, cfg):
        super(TCTRN_Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        decoder_param = cfg["TCTRN"]["DECODER"]

        self.convolutions.append(
            nn.Sequential(
                TCTRN_Conv1d(
                    in_channels=cfg["AUDIO"]["N_MEL_CHANNELS"],
                    out_channels=decoder_param["POSTNET_EMBD_DIM"],
                    kernel_size=decoder_param["POSTNET_KERNEL_SIZE"],
                    padding=int((decoder_param["POSTNET_KERNEL_SIZE"]-1)/2),
                    dilation=1,
                    init_gain="tanh"
                ),
                nn.BatchNorm1d(decoder_param["POSTNET_EMBD_DIM"])
            )
        )

        for i in range(1, decoder_param["POSTNET_N_CONV"]-1):
            self.convolutions.append(
                nn.Sequential(
                    TCTRN_Conv1d(
                        in_channels=decoder_param["POSTNET_EMBD_DIM"],
                        out_channels=decoder_param["POSTNET_EMBD_DIM"],
                        kernel_size=int((decoder_param["POSTNET_KERNEL_SIZE"] - 1) / 2),
                        dilation=1,
                        init_gain="tanh"
                    ),
                    nn.BatchNorm1d(decoder_param["POSTNET_EMBD_DIM"])
                )
            )

        self.convolutions.append(
            nn.Sequential(
                TCTRN_Conv1d(
                    in_channels=decoder_param["POSTNET_EMBD_DIM"],
                    out_channels=cfg["AUDIO"]["N_MEL_CHANNELS"],
                    padding=int((decoder_param["POSTNET_EMBD_DIM"]-1)/2),
                    dilation=1,
                    init_gain="tanh"
                ),
                nn.BatchNorm1d(cfg["AUDIO"]["N_MEL_CHANNELS"])
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) -1):
            x = F.dropout(
                torch.tanh(self.convolutions[i](x)),
                p=0.5,
                training=self.training
            )
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x


class TCTRN_Decoder(nn.Module):
    def __init__(self, cfg):
        super(TCTRN_Decoder, self).__init__()
        # param list
        self.n_mel_channels = cfg["AUDIO"]["N_MEL_CHANNELS"]

        encoder_param = cfg["TCTRN"]["ENCODER"]
        self.encoder_embedding_dim = encoder_param["EMBD_DIM"]

        decoder_param = cfg["TCTRN"]["DECODER"]
        self.n_frames_per_step = decoder_param["N_FRAMES_PER_STEP"]
        self.attention_rnn_dim = decoder_param["ATTENTION_RNN_DIM"]
        self.decoder_attention_dim = decoder_param["ATTENTION_DIM"]
        self.decoder_rnn_dim = decoder_param["RNN_DIM"]
        self.prenet_dim = decoder_param["PRENET_DIM"]
        self.max_decoder_steps = decoder_param["MAX_STEPS"]
        self.gate_threshold = decoder_param["GATE_THRESHOLD"]
        self.p_attention_dropout = decoder_param["ATTENTION_DROPOUT"]
        self.p_decoder_dropout = decoder_param["DROPOUT"]

        self.location_inner_dim = decoder_param["LOCATION_INNER_DIM"]
        self.location_kernel_size = decoder_param["LOCATION_KERNEL_SIZE"]

        # 2 linear layers: n_mel_channels*n_frames_per_step -> prenet_dim -> prenet_dim
        self.prenet = TCTRN_Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim]
        )

        # only a single cell, prenet_dim+encoder_embedding_dim -> attention_rnn_dim
        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim,
            self.attention_rnn_dim
        )

        self.attention_layer = TCTRN_Attention(
            attention_rnn_dim=self.attention_rnn_dim,
            embedding_dim=self.encoder_embedding_dim,
            attention_dim=self.decoder_attention_dim,
            location_inner_dim=self.location_inner_dim,
            location_kernel_size=self.location_kernel_size
        )

        self.decoder_rnn = nn.LSTMCell(
            input_size=self.decoder_rnn_dim + self.encoder_embedding_dim,
            hidden_size=self.decoder_rnn_dim,
            bias=True
        )

        self.linear_projection = TCTRN_Linear(
            in_dim=self.decoder_rnn_dim + self.encoder_embedding_dim,
            out_dim=self.n_mel_channels * self.n_frames_per_step)

        self.gate_layer = TCTRN_Linear(
            in_dim=self.decoder_rnn_dim + self.encoder_embedding_dim,
            out_dim=1,
            bias=True,
            init_gain="sigmoid"
        )

    def get_go_frame(self, memory):
        """init the first start frame to feed into decoder
        memory: encoder output, shape (B, max_time, embedding_dim)
        RETURN
        shape (B, n_mel_channels*n_frames_per_step)
        """
        batch_size = memory.size()[0]
        start_frame = memory.new_zeros(
            (batch_size, self.n_mel_channels * self.n_frames_per_step),
            requires_grad=True
        )
        return start_frame

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size()[0]
        MAX_TIME = memory.size()[1]

        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask
        )

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1
        )
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context
        )

        gated_prediction = self.gate_layer(
            decoder_hidden_attention_context,
        )
        return decoder_output, gated_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_output += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            # TODO Squeeze?
            gate_output += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )
        return mel_outputs, gate_outputs, alignments
