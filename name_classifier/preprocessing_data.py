import string
import unicodedata
import glob
import os


# 获取所有英文大小写字符和常用标点
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'
all_letters = string.ascii_letters + " .,;'"
# 获取所有all_letters 长度  57
n_letters = len(all_letters)


# 非英文外的语言,会出现重音等非规范化字母,进行规范化处理 如: Ślusàrski ---> Slusarski
def unicodeToAscii(s):
    """
    :param s: 原始字符串, 即人名
    :return:
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn'
                   and c in all_letters)


# 读取文件内容，并对文件中每一个名字进行规范化处理
def readLines(file_name):
    """
    :param file_name: 文件路径
    :return:
    """
    # read()将文件内容一次读取, 每行为一个人名, 行与行间以\n进行分隔
    # readline() 按行读取,每次读取一行即遇到\n结束读取
    # readlines() 一次性读取文件内容, 并返回list
    # readlines() 以生成器方式读取
    # [unicodeToAscii(line) for line in open(file_name, encoding = 'utf-8').readlines()]
    return [unicodeToAscii(line) for line in open(file_name, encoding = 'utf-8').read().strip().split('\n')]


def build_dataset():
    # {'chinese' : [name1, name2..], 'english' : [name1, name2...]...}
    category_lines = {}
    # ['chinese', 'english'...]
    all_categories = []
    data_path = './data/names/'

    # data下每个文件名字为category, 文件内容为相应的人名
    for file_path in glob.glob(data_path + '*.txt'):
        # 获取每个类别 file_path为 './data/names/chinese.txt'具体路径
        # os.path.basename(dir)  ['chinese.txt', 'english.txt'...]
        category = os.path.basename(file_path).split('.')[0]
        # 构建字典需要获取每个文件的内容即人名列表
        lines = readLines(file_path)
        category_lines[category] = lines
        all_categories.append(category)

    return category_lines, all_categories


if __name__ == '__main__':
    print(all_letters)
    print(n_letters)
    s = "Ślusàrski"
    print('Ślusàrski->', unicodeToAscii(s))
    res = readLines('data/names/Spanish.txt')
    print(res[:10])

    category_lines, all_category = build_dataset()
    print('category_lines', list(category_lines.items())[0])
    print('all_category', all_category[:5])
    print('类别总数为:' , len(all_category))