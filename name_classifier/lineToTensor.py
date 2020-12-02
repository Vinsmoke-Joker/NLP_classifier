import torch
from preprocessing_data import build_dataset, n_letters, all_letters

category_lines, all_category = build_dataset()


# 对每个名字进行one-hot编码,并封装为tensor

def lineToTensor(line):
    """
    :param line: 每个名字
    :return:
    """
    # tensor的形状为[len(line), 1, n_letters]
    # 表示一个batch为一个名字的所有字母个数,1行, n_letters列,每行为一个字母的one-hot
    # 由于规范化后都为英文字母和标点，n_letters即为词表大小
    tensor = torch.zeros(len(line), 1, n_letters)

    for idx, char in enumerate(line):
        # one-hot编码中, 每行表示一个字母char的one-hot,列为char的值为1
        tensor[idx][0][all_letters.index(char)] = 1
        # 使用字符串方法find找到每个字符在all_letters中的索引
        # tensor[idx][0][all_letters.find(char)]
    return tensor


if __name__ == '__main__':
    line = 'bai'
    print(lineToTensor(line))
