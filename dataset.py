import torch
from save_vocab import get_x_and_y, FILE_PATHS

# 定义数据集类

class TextDataSet(torch.utils.data.Dataset):
    def __init__(self, file_type, vocab_path, max_len=32):
        # 调用get_x_and_y函数获取数据
        self.x, self.y = get_x_and_y(file_type)

        # 读取词表
        self.words, self.word_to_idx = read_vocab(vocab_path)

        # 将数据padding到固定长度
        self.x = padding_x(self.x, max_len)
        
        # 将数据转换为idx
        self.x = x_to_idx(self.x, self.word_to_idx)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 读取保存的词表，并将其转化为 词的列表 和 word到idx的字典
def read_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        words = file.read()
    words = words.splitlines()
    word_to_idx = {}
    for i, word in enumerate(words):
        word_to_idx[word] = i
    return words, word_to_idx

# 将idx转换为原始的数据
def idx_to_words(idx, words):
    return "".join([words[i] for i in idx])

# 将数据padding到固定长度，不足的用"<pad>"填充
# 模型的输入数据需要是固定长度的，所以需要将数据padding到固定长度
def padding_x(xs, max_len=32):
    padding_x = []
    for x in xs:
        if len(x) < max_len:
            padding_x.append(list(x) + ["<pad>"] * (max_len - len(x)))
        else:
            padding_x.append(x[:max_len])
    return padding_x

# 将一条数据转换为idx
def x_to_idx(x, word_to_idx):
    return torch.tensor([[word_to_idx[word] for word in sentence] for sentence in x])
