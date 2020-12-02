import torch
import torch.nn as nn
from preprocessing_data import n_letters, build_dataset


category_lines, all_category = build_dataset()


class GRU_Model(nn.Module):
    def __init__(self):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(
            input_size = n_letters,
            hidden_size = 128,
            num_layers = 1,
            batch_first = True,
            bidirectional = False
        )
        self.fc = nn.Linear(128, len(all_category))
        self.log_softmax = nn.LogSoftmax(dim = -1)


    def forward(self, x, hidden):
        output, hidden = self.gru(x.unsqueeze(0), hidden)
        return self.log_softmax(self.fc(output)), hidden


    def init_hidden(self):
        return torch.zeros(1, 1, 128)