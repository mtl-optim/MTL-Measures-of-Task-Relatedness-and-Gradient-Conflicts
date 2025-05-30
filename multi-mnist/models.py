import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLeNetShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(320, 50)
        
    def generate_dropout_mask(self, x, p=0.5):
        if self.training:
            mask = torch.bernoulli(torch.full((1, x.size(1), 1, 1), 1 - p, device=x.device))
        else:
            mask = torch.ones(1, x.size(1), 1, 1, device=x.device)
        return mask.expand_as(x)
    
    def forward(self, x, mask=None):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        if mask is None:
            mask = self.generate_dropout_mask(x)
        if self.training:
            x = x * mask
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        return x, mask

class MultiLeNetTaskSpecific(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, mask=None):
        x = F.relu(self.fc1(x))
        if mask is None:
            mask = torch.bernoulli(torch.full_like(x, 0.5))
        if self.training:
            x = x * mask
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), mask