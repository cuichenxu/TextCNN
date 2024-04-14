from model import TextCNN
import torch
from dataset import read_vocab, idx_to_words
from save_vocab import get_class

vocab_path = "path of vacab.txt"
class_path = "path of class.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes, class_to_idx = get_class(class_path)
words, word_to_idx = read_vocab(vocab_path)

inputs = "词汇阅读是关键 08年考研暑期英语复习全指南"
label = 3
inputs = list(inputs.replace(' ', ''))
inputs = [word_to_idx[word] for word in inputs]
inputs += [word_to_idx["<pad>"]] * (32 - len(inputs))
inputs = torch.tensor(inputs).to(device)
inputs = inputs.unsqueeze(0)

model = TextCNN(len(words), 128, len(classes), 1).to(device)
model.load_state_dict(torch.load("/home/alien/dir1/TextCNN/save_model/dataloader_model_4.pth"))
model.eval()

with torch.no_grad():
    pred_y = model(inputs)
    _, predicted = torch.max(pred_y.data, 1)
    print(f"predict:{classes[predicted.item()]}")
    print(f"true:{classes[label]}")
    