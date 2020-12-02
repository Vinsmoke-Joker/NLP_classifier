import torch.nn as nn
from preprocessing_data import n_letters, build_dataset
import torch


category_lines, all_category = build_dataset()


class RNN_Model(nn.Module):
    def __init__(self):
        super(RNN_Model, self).__init__()
        # 由于已经进行one-hot处理,不需要再embedding
        self.rnn = nn.RNN(
            input_size = n_letters,
            hidden_size = 128,
            num_layers = 1,
            batch_first = True,
            bidirectional = False
        )
        self.fc = nn.Linear(128, len(all_category))
        self.log_softmax = nn.LogSoftmax(dim = -1)


    def forward(self, x, hidden):
        # 进行训练时, 需要将人名中的每一个字母输入网络进行训练
        # input_tensor [len(line), 1, n_letters]
        # 第i个字母为input_tensor[i], 形状为[1, n_letters] 即为x
        # rnn输入形状要求为3维度, 因此需要对x进行维度扩展
        # 扩展为[1, 1, n_letters]
        x = x.unsqueeze(0)
        output, hidden = self.rnn(x, hidden)
        # output [1, 1, 128]
        # hidden [1, 1, 128]
        output = self.fc(output)
        # output [1, 1, 18]
        output = self.log_softmax(output)
        return output, hidden


    def init_hidden(self):
        return torch.zeros(1, 1, 128)