from model import TextCNN
import torch
from dataset import read_vocab, idx_to_words
from save_vocab import get_class

vocab_path = "path of vacab.txt"
class_path = "path of class.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取词表和类别
classes, class_to_idx = get_class(class_path)
words, word_to_idx = read_vocab(vocab_path)

# 输入数据和标签
inputs = "词汇阅读是关键 08年考研暑期英语复习全指南"
label = 3

# 将输入数据按照字划分，并转换为idx
inputs = list(inputs.replace(' ', ''))
inputs = [word_to_idx[word] for word in inputs]

# 将输入数据padding到固定长度
inputs += [word_to_idx["<pad>"]] * (32 - len(inputs))

inputs = torch.tensor(inputs).to(device)

# 使用unsqueeze对输入的数据升维
# 将输入数据转换为(batch_size, max_len)的形状
inputs = inputs.unsqueeze(0)

# 加载模型
model = TextCNN(len(words), 128, len(classes), 1).to(device)
model.load_state_dict(torch.load("/home/alien/dir1/TextCNN/save_model/dataloader_model_4.pth"))
model.eval()

with torch.no_grad():
    pred_y = model(inputs)
    _, predicted = torch.max(pred_y.data, 1)
    print(f"predict:{classes[predicted.item()]}")
    print(f"true:{classes[label]}")
    