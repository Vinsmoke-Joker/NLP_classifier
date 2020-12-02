import torch
import torch.nn as nn
from preprocessing_data import  build_dataset, n_letters


category_lines, all_category = build_dataset()


class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size = n_letters,
            hidden_size = 128,
            num_layers = 1,
            batch_first = True,
            bidirectional = False
        )
        self.fc = nn.Linear(128, len(all_category))
        self.log_softmax = nn.LogSoftmax(dim = -1)


    def forward(self, x, hidden_and_c):
        output, hidden_and_c = self.lstm(x.unsqueeze(0), hidden_and_c)
        return self.log_softmax(self.fc(output)), hidden_and_c


    def init_hidden(self):
        # (hidden, c)
        return (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))