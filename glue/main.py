import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from models import MultiLeNetShared, MultiLeNetTaskSpecific
from utils import *
import sys
from torch.utils.data import DataLoader

seed = 70

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

method = sys.argv[1]
num_epochs = int(sys.argv[2])
available_methods = ['pcgrad', 'classical', 'modifiedgd', 'graddrop', 'cagrad', 'mgd']

if method not in available_methods:
    raise ValueError(f"Error: The method '{method}' is not available. Choose from {available_methods}")
else:
    print(f"Method selected: {method}")
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
PATH = '/home/glebn/ner/multi_mnist/Pytorch-PCGrad/data/Data/MultiMNIST'

accuracy = lambda logits, gt: ((logits.argmax(dim=-1) == gt).float()).mean()
to_dev = lambda inp, dev: [x.to(dev) for x in inp]

transformer = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))])

train_dst = MultiMNIST(PATH, train=True, transform=transformer, multi=True)
val_dst = MultiMNIST(PATH, train=False, transform=transformer, multi=True)

train_loader = DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=1)

nets = {
    'shared': MultiLeNetShared().to(device),
    'left': MultiLeNetTaskSpecific().to(device),
    'right': MultiLeNetTaskSpecific().to(device)
}

param = [p for v in nets.values() for p in list(v.parameters())]
optimizer = torch.optim.Adam(param, lr=0.0005)

rng = np.random.default_rng()
grad_dims = []
for name, param in nets['shared'].named_parameters():
    num_params = param.data.numel()
    grad_dims.append(param.data.numel())
grads = torch.Tensor(sum(grad_dims), 2).cuda()

for ep in range(num_epochs):
    nets['shared'].train()
    nets['left'].train()
    nets['right'].train()
    for batch in train_loader:
        mask = None
        
        img, label_l, label_r = to_dev(batch, device)
        out_shared, mask = nets['shared'](img, mask)
        out_l, mask_l = nets['left'](out_shared, None)
        out_r, mask_r = nets['right'](out_shared, None)

        losses = [F.nll_loss(out_l, label_l), F.nll_loss(out_r, label_r)]
        
        optimizer.zero_grad()
        if method == "pcgrad":
            for i in range(2):
                losses[i].backward(retain_graph=True)
                grad2vec(nets['shared'], grads, grad_dims, i)
                zero_grad_shared_modules(nets['shared'])
            g = pcgrad(grads, rng)
            overwrite_grad(nets['shared'], g, grad_dims)
        elif method == "cagrad":
            for i in range(2):
                losses[i].backward(retain_graph=True)
                grad2vec(nets['shared'], grads, grad_dims, i)
                zero_grad_shared_modules(nets['shared'])
            g = cagrad(grads)
            overwrite_grad(nets['shared'], g, grad_dims)
        elif method == "modifiedgd":
            for i in range(2):
                losses[i].backward(retain_graph=True)
                grad2vec(nets['shared'], grads, grad_dims, i)
                zero_grad_shared_modules(nets['shared'])
            g = modifiedgd(grads)
            overwrite_grad(nets['shared'], g, grad_dims)
        elif method == "mgd":
            for i in range(2):
                losses[i].backward(retain_graph=True)
                grad2vec(nets['shared'], grads, grad_dims, i)
                zero_grad_shared_modules(nets['shared'])
            g = mgd(grads)
            overwrite_grad(nets['shared'], g, grad_dims)
        elif method == "graddrop":
            for i in range(2):
                losses[i].backward(retain_graph=True)
                grad2vec(nets['shared'], grads, grad_dims, i)
                zero_grad_shared_modules(nets['shared'])
            g = graddrop(grads)
            overwrite_grad(nets['shared'], g, grad_dims)
        elif method == "classical":
            loss_ = (losses[0]+losses[1])
            loss_.backward()
        optimizer.step()

    losses, acc = [], []
    nets['shared'].eval()
    nets['left'].eval()
    nets['right'].eval()
    for batch in val_loader:
        img, label_l, label_r = batch
        img = img.to(device)
        label_l = label_l.to(device)
        label_r = label_r.to(device)
        out_shared, mask = nets['shared'](img, None)
        out_l, mask_l = nets['left'](out_shared, None)
        out_r, mask_r = nets['right'](out_shared, None)

        losses.append([
            F.nll_loss(out_l, label_l).item(),
            F.nll_loss(out_r, label_r).item()
        ])
        acc.append(
            [accuracy(out_l, label_l).item(),
            accuracy(out_r, label_r).item()])

    losses, acc = np.array(losses), np.array(acc)
    print('[Test] epoch {}/{}: accuracy (left, right) = {:5.3f}, {:5.3f}'.format(
        ep, num_epochs, acc[:,0].mean(), acc[:,1].mean()))
    
