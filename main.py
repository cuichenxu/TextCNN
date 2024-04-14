from save_vocab import save_vocab, get_class
from dataset import TextDataSet, idx_to_words, read_vocab
from train import train
import torch
from model import TextCNN

vocab_path = "path of vacab.txt"
class_path = "path of class.txt"

# save_vocab(vocab_path)

dataset = TextDataSet("train", vocab_path)
dev_dataset = TextDataSet("dev", vocab_path)
test_dataset = TextDataSet("test", vocab_path)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=4, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes, class_to_idx = get_class(class_path)
words, word_to_idx = read_vocab(vocab_path)

vocab_size = len(words)
class_num = len(classes)
embedding_dim = 128

model = TextCNN(vocab_size, embedding_dim, class_num, batch_size=4)
model = model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
epoches = 10

train(dataloader, dev_dataloader, model, loss_fn, optimizer, epoches, device)