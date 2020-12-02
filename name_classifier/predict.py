import torch
from RNN_model import RNN_Model
from LSTM_model import LSTM_Model
from GRU_model import GRU_Model
from lineToTensor import lineToTensor
from preprocessing_data import build_dataset


def evaluate_RNN(line_tensor):
    """
    RNN 评估
    :param line_tensor: 名字张量 [len(line), 1, n_letters]
    :return:
    """
    model = RNN_Model()
    model.load_state_dict(torch.load('./model/RNN_model.pkl'))
    hidden = model.init_hidden()
    for i in range(line_tensor.size(0)):
        # line_tensor[len(line), 1, n_letters]
        # 训练时,for i in range(len(line)) ->line_tensor[i]  [1, n_letters],n_letters也是词向量长度(one-hot)
        # 训练后, [1, 1, hidden_size] ->fc [1, 1, categories]
        # output [1, 1, 18]
        # hidden [1, 1, 128]
        output, hidden = model(line_tensor[i], hidden)
    return output


def evaluate_LSTM(line_tensor):
    model = LSTM_Model()
    model.load_state_dict(torch.load('./model/LSTM_model.pkl'))
    hidden = model.init_hidden()
    for i in range(line_tensor.size(0)):
        output, hidden = model(line_tensor[i], hidden)
    return output


def  evaluate_GRU(line_tensor):
    model = GRU_Model()
    model.load_state_dict(torch.load('./model/GRU_model.pkl'))
    hidden = model.init_hidden()
    for i in range(line_tensor.size(0)):
        output, hidden = model(line_tensor[i], hidden)
    return output


def predict(input_line, evaluate, n_predictions = 3):
    """
    对输入的名字进行分类预测
    :param input_line: 输入的名字,字符串
    :param evaluate: 评估函数
    :param n_predictions: 取最有可能的三个
    :return:
    """
    print('\n >>> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        # output [1, 1, 18]
        # torch.topk(input, k, dim=0, largest=True, sorted=True, out=None)
        # 在列即类别的维度上求最大值的索引
        # value和index的维度与输入即output维度一致
        value, index = torch.topk(output, k = n_predictions, dim = -1, sorted = True)
        # 保存结果列表
        predictions = []
        # 获取所有类别
        # category_lines {'english':[name1, name2...]}
        # all_category [english, china...]
        category_lines, all_category = build_dataset()
        # 取出对应类别,并添加进predictions中
        for i in range(n_predictions):
            val = value[0][0][i].item()
            idx = index[0][0][i].item()
            category = all_category[idx]
            predictions.append([val, category])
            print('-->({:.4f}){} '.format(val, category))

if __name__ == '__main__':
    model_dict = {'RNN' : evaluate_RNN, 'LSTM' : evaluate_LSTM, 'GRU' :evaluate_GRU}
    for model_type, evaluate_fn in model_dict.items():
        print('-' * 50)
        print(model_type)
        predict('Dovesky', evaluate_fn)
        predict('Jackson', evaluate_fn)
        predict('Satoshi', evaluate_fn)
