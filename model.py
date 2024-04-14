import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_num, batch_size):
        super(TextCNN, self).__init__()
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv2d(1, 512, (3, embedding_dim))
        self.conv2 = nn.Conv2d(1, 512, (4, embedding_dim))
        self.conv3 = nn.Conv2d(1, 512, (5, embedding_dim))
        self.fc = nn.Linear(512 * 3, class_num)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def pooling(self, x):
        x = x.squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(-1))
        return x
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(self.batch_size, 1, 32, 128)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x3 = self.pooling(x3)
        x = torch.cat((x1, x2, x3), 2)
        x = x.reshape(self.batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
