from preprocessing_data import n_letters, build_dataset
import random
import torch
from lineToTensor import lineToTensor
import time
import math

category_lines, all_category = build_dataset()

def random_data():
    # 随机选择一个类别，并挑选该类别中的一个名字,封装为tensor返回
    category = random.choice(all_category)
    line = random.choice(category_lines[category])
    category_tensor = torch.LongTensor([all_category.index(category)])
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def timeSince(since):
    """
    计算时间函数
    :param since: 当前时间
    :return:
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s = s - m*60
    return '%d m : %d s' % (m, s)





if __name__ == '__main__':
    category, line, category_tensor, line_tensor = random_data()
    print('category', category)
    print('line', line)
    print('category_tensor', category_tensor)
    print('category_tensor.size', category_tensor.size())
    print('line_tensor', line_tensor)
    print('line_tensor.size', line_tensor.size())