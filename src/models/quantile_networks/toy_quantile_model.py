import torch
import torch.nn.functional as F


class Net_common(torch.nn.Module):
    def __init__(self):
        super(Net_common, self).__init__()
        self.ln1 = torch.nn.Linear(5, 32)
        self.ln2 = torch.nn.Linear(32, 64)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        return x


class FC(torch.nn.Module):
    def __init__(self, n_feature):
        super(FC, self).__init__()
        self.ln1 = torch.nn.Linear(n_feature, 1)

    def forward(self, x):
        x = self.ln1(x)
        return x
