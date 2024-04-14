import torch
from save_vocab import get_x_and_y, FILE_PATHS

class TextDataSet(torch.utils.data.Dataset):
    def __init__(self, file_type, vocab_path, max_len=32):
        self.x, self.y = get_x_and_y(file_type)
        self.words, self.word_to_idx = read_vocab(vocab_path)
        self.x = padding_x(self.x, max_len)
        self.x = x_to_idx(self.x, self.word_to_idx)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def read_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        words = file.read()
    words = words.splitlines()
    word_to_idx = {}
    for i, word in enumerate(words):
        word_to_idx[word] = i
    return words, word_to_idx

def idx_to_words(idx, words):
    return "".join([words[i] for i in idx])

def padding_x(xs, max_len=32):
    padding_x = []
    for x in xs:
        if len(x) < max_len:
            padding_x.append(list(x) + ["<pad>"] * (max_len - len(x)))
        else:
            padding_x.append(x[:max_len])
    return padding_x

def x_to_idx(x, word_to_idx):
    return torch.tensor([[word_to_idx[word] for word in sentence] for sentence in x])
