import matplotlib.pyplot as plt
import re
import os

# 读取文件中保存的损失并提取文件名中的耗时
def get_fig_data():
    data_path = './figure'
    # 获取文件名列表
    file_names = os.listdir(data_path)

    # 构造字典data_dict{RNN : [time1, [loss1, loss2]], GRU:[time2, [loss1, loss2]], LSTM: [time3, [loss1, loss2]]}
    data_dict = {}

    for name in file_names:
        temp = []
        # 将GRU45.45464.txt -> GRU45 方便后续正则匹配
        time = name.split('.')[0]
        temp.append(int(''.join(re.findall(r'\d',time))))
        # 进行损失值的读取
        path = os.path.join(data_path, name)
        with open(path, encoding = 'utf-8') as f:
            loss_list = [float(i) for i in f.readlines()]
        temp.append(loss_list)
        # findall 匹配成功后返回每个字符结果列表
        data_dict[''.join(re.findall(r'\D', time))]= temp
    return data_dict


def plot_figure(loss_list1, time1, loss_list2, time2, loss_list3, time3):
    plt.figure(0)
    plt.plot(loss_list1, label = 'rnn')
    plt.plot(loss_list2, 'r', label = 'lstm')
    plt.plot(loss_list3, 'b', label = 'gru')
    plt.title('LOSS')
    plt.legend(loc = 'upper right')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.savefig('./loss.png')

    plt.figure(1)
    x_data = ['RNN', 'LSTM', 'GRU']
    y_data = [time1, time2, time3]
    plt.bar(range(len(x_data)), y_data, tick_label = x_data)
    plt.title('Cost Time')
    plt.savefig('./time.png')


if __name__ == '__main__':
    data_dict = get_fig_data()
    time1, loss_list1 = data_dict['RNN'][0], data_dict['RNN'][1]
    time2, loss_list2 = data_dict['LSTM'][0], data_dict['LSTM'][1]
    time3, loss_list3 = data_dict['GRU'][0], data_dict['GRU'][1]
    plot_figure(loss_list1, time1, loss_list2, time2, loss_list3, time3)