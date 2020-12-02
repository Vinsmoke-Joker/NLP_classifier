import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing_data import n_letters, build_dataset
import os
import argparse
from RNN_model import RNN_Model
from LSTM_model import LSTM_Model
from GRU_model import GRU_Model
import time
from random_data import random_data, timeSince, all_category


parser = argparse.ArgumentParser(description = 'Name_Classifier_Task')
parser.add_argument('--model', type = str, required = False, help = 'Choose a model : RNN, LSTM, GRU')
args = parser.parse_args()

model_path = './model/' + args.model + '_model.pkl'
print(model_path)
# 网络中进行了log_softmax 因此计算损失采用NLLLoss
criterion = nn.NLLLoss()
# 设置学习率
learning_rate = 0.005


if args.model == 'RNN':
    model = RNN_Model()
elif  args.model == 'LSTM':
    model = LSTM_Model()
else:
    model = GRU_Model()

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))


def train_rnn(category_tensor, line_tensor):
    """
    :param category_tensor: 转换后的类别tensor,即目标值(随机抽取的一个)
    :param line_tensor: 进行one-hot和tensor的封装的一个人名(随机抽取类别中随机抽取的一个人名)
    :return:
    """
    # 每次训练batch为一个完整的人名长度 line_tensor[len(name), 1, n_letters]
    # 每次将人名中每个字母放入网络进行训练 line_tensor[i]  [1, n_letters]
    hidden = model.init_hidden()
    model.zero_grad()
    for i in range(line_tensor.size(0)):
        output, hidden = model(line_tensor[i], hidden)
    # output 为人名中每个字母,要计算整个人名的损失需要模型更新完output
    # output [1, 1, hidden_size]
    # category_tensor [1]
    # 计算loss,需要对output 进行降维
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    # 显示更新参数
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def train_lstm(category, line_tensor):
    """
    :param category:
    :param line_tensor:
    :return:
    """
    model.zero_grad()
    hidden = model.init_hidden()
    for i in range(line_tensor.size(0)):
        output, hidden_c = model(line_tensor[i], hidden)
    loss = criterion(output.squeeze(0), category)
    loss.backward()
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()


def train_gru(category_tensor, line_tensor):
    model.zero_grad()
    hidden = model.init_hidden()
    for i in range(line_tensor.size(0)):
        output, hidden = model(line_tensor[i], hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    for p in model.parameters():
        p.data.sub_(learning_rate, p.grad.data)
    return output, loss.item()


def train_type_fn(category_tensor, line_tensor):
    if args.model == 'RNN':
        output, loss = train_rnn(category_tensor, line_tensor)
    elif args.model == 'LSTM':
        output, loss = train_lstm(category_tensor, line_tensor)
    else:
        output, loss = train_gru(category_tensor, line_tensor)
    return output, loss


def train():
    """
    训练函数,打印结果保存模型并绘制图像
    :param train_type_fn: RNN / LSTM / GRU 某种训练方式
    :return:
    """
    # 设置迭代次数
    n_iters = 50000
    # 设置打印间隔
    print_every = 50
    # 图像取点间隔
    plot_every = 10
    # 收集损失
    loss_list = []
    # 当前时间
    since = time.time()
    # 设置初始间隔损失
    current_loss = 0

    for iter in range(1, n_iters + 1):
        # 获取随时生成的数据
        category, line, category_tensor, line_tensor = random_data()
        # 将数据放入网络进行训练,获取损失和输出
        output, loss = train_type_fn(category_tensor, line_tensor)
        current_loss += loss

        # 打印间隔
        if iter % print_every == 0:
            # 同时获取当前预测,
            value, index = torch.max(output, dim = -1)
            pred_index = index.squeeze().item()
            pred = all_category[pred_index]
            correct = '√' + (category) if pred == category else '×(%s)' % category
            print('iter:{}/{:.2f}%, time:{}, loss:{:.4f}, line:{}, pred:{}, correct:{}'.format(
                iter, iter/ n_iters * 100, timeSince(since), loss, line, pred, correct
            ))
        # 收集绘图点
        if iter % plot_every == 0:
            loss_list.append(current_loss / plot_every)
            current_loss = 0
        if iter % 100 == 0:
            torch.save(model.state_dict(), model_path)
    return loss_list, time.time() - since


def save_loss(loss_list, time):
    """
    收集损失和时间，用于绘图
    :param loss_list: 训练得到的loss_list
    :param time:  训练收集的耗时
    :return:
    """
    model_name = args.model
    # 将所耗费时间写入文件名中
    with open('./figure/' + model_name + str(time) + '.txt', encoding = 'utf-8', mode = 'a') as f:
        # 读取每个损失, 并转换字符串 拼接\n,文件中每行为一个损失值
         f.write(''.join([str(i) + '\n' for i in loss_list]))


if __name__ == '__main__':
    loss_list, time = train()
    save_loss(loss_list, time)



    