
# model

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRU(nn.Module):

    def __init__(self, vocab, dimension=128, sequenceDepth = 64, dropoutWithinLayers = 0.3, dropoutOutput = 0.3):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(len(vocab), sequenceDepth)
        self.dimension = dimension
        self.GRU = nn.GRU(input_size=sequenceDepth,
                            hidden_size=dimension,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropoutWithinLayers)
        self.dropOutput = nn.Dropout(p=dropoutOutput)

        # output layer
        self.fc = nn.Linear(2 * dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.GRU(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.dropOutput(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out