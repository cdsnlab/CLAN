import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, dimension=128, num_classes = 4, feature_dim = 7 , dim =128, hidden_dim=64, depth=12, kernel_size=10, stride =3,):
        super(LSTM, self).__init__()
        super().__init__()
        self.embeddings = nn.Embedding(10000, dim)
        self.lstm = nn.LSTM(dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data, label):

        #text_emb = self.embedding(text)
        x = self.embeddings(data)
        x = self.drop(x)
        output, (ht, ct) = self.lstm(x)

        return self.linear(ht[-1])

class Net(nn.Module):
    def __init__(self, num_classes, feature_dim, dim, kernel_size=10, stride = 10):
        super(Net, self).__init__()
        self.to_patch_embedding = nn.Sequential(nn.Conv1d(feature_dim, dim, kernel_size=kernel_size, stride =stride))
        self.dim = dim
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(0.)

    def forward(self, x):              
        x = torch.transpose(x,1,2).cuda()
        x = self.to_patch_embedding(x).cuda()
        x = torch.transpose(x,1,2).cuda()
        x = F.relu(F.max_pool1d(x, 1)).cuda()
        b, n, d = x.shape

        # image input을 펼쳐준다.
        x = torch.reshape(x,(-1, n*d)).cuda()
        
        self.fc1 = nn.Linear(n*d, self.dim).cuda()
        # 은닉층을 추가하고 활성화 함수로 relu 사용
        x = F.relu(self.fc1(x)).cuda()
        x = self.dropout(x).cuda()

        # 은닉층을 추가하고 활성화 함수로 relu 사용
        x = F.relu(self.fc2(x)).cuda()
        x = self.dropout(x).cuda()

        # 출력층 추가
        x = self.fc3(x).cuda()
        return x