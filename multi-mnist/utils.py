import torch
import numpy as np
from copy import deepcopy
import torch.utils.data as data
from PIL import Image
import os
import os.path
from min_norm_solvers import MinNormSolver
from scipy.optimize import minimize_scalar

condition_numbers = []

def grad2vec(m, grads, grad_dims, task):
    grads[:, task].fill_(0.0)
    cnt = 0
    for name, param in m.named_parameters():
        grad = param.grad
        if grad is not None:
            grad_cur = grad.data.detach().clone()
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, task].copy_(grad_cur.data.view(-1))
        cnt += 1
        
def zero_grad_shared_modules(m):
    for module in m.modules():
        module.zero_grad()

def pcgrad(grads, rng):
    grad_vec = grads.t()
    num_tasks = 2

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (
        grad_vec.norm(dim=1, keepdim=True) + 1e-8
    )  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[
            task_indices
        ]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(
            dim=1, keepdim=True
        )  # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g

def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
    U = torch.rand_like(grads[:,0])
    M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g

def mgd(grads):
    grads_cpu = grads.t().cpu()
    sol, min_norm = MinNormSolver.find_min_norm_element([
        grads_cpu[t] for t in range(grads.shape[-1])])
    w = torch.FloatTensor(sol).to(grads.device)
    g = grads.mm(w.view(-1, 1)).view(-1)
    return g

def cagrad(grads, alpha=0.5, rescale=0):
    g1 = grads[:,0]
    g2 = grads[:,1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12)

    # want to minimize g_w^Tg_0 + c*g_0*g_w
    coef = alpha * g0_norm
    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-8) + 0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-8)
    lmbda = coef / (gw_norm+1e-8)
    g = (0.5+lmbda*x) * g1 + (0.5+lmbda*(1-x)) * g2 # g0 + lmbda*gw
    if rescale== 0:
        return g
    elif rescale== 1:
        return g / (1+alpha**2)
    else:
        return g / (1 + alpha)

def modifiedgd(grads, eps=1e-5):
    g1 = grads[:,0]
    g2 = grads[:,1]

    grad_vec = grads.t()

    norm_g1 = torch.norm(g1, 2)
    norm_g2 = torch.norm(g2, 2)
    tensor_norms = [norm_g1, norm_g2]
    min_norm = min(norm_g1, norm_g2)
    diff_between_grads = abs(norm_g1 - norm_g2)

    pair = torch.stack([g1, g2])
    U, singular_values, Vh = torch.linalg.svd(pair, full_matrices=False)
    # singular_values = np.linalg.svd(pair.cpu().numpy(), compute_uv=False)

    if singular_values.min() <= eps:
        cond_num = -1
    else:
        cond_num = singular_values.max() / singular_values.min()
        condition_numbers.append(cond_num.cpu())
        
    # cond_threshold = np.percentile(condition_numbers, 70)

    if len(condition_numbers) == 0 or cond_num == -1 or cond_num < np.percentile(condition_numbers, 70): #:
        g = grad_vec.mean(dim=0)
        return g
    else:
        scaled_grads = []
        for grad, norm in zip(grad_vec, tensor_norms):
            if norm > 0:
                scaling_factor = min_norm / norm
                scaled_grads.append(grad * scaling_factor)
            else:
                scaled_grads.append(grad)
        avg_grad = torch.stack(scaled_grads).mean(dim=0)
        return avg_grad
    
def overwrite_grad(m, newgrad, grad_dims):
    newgrad = newgrad * 2
    cnt = 0
    for name, param in m.named_parameters():
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = newgrad[beg: en].contiguous().view(param.data.size())
        param.grad = this_grad.data.clone()
        cnt += 1
        
class MultiMNIST(data.Dataset):
    processed_folder = 'processed'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, multi=False):

        self._root = os.path.expanduser(root)
        self._transform = transform
        self._train = train
        self._multi = multi

        if self._train:
            self.train_data, self.train_labels_l, self.train_labels_r = torch.load(
                os.path.join(self._root, self.processed_folder, self.multi_training_file))
        else:
            self.test_data, self.test_labels_l, self.test_labels_r = torch.load(
                os.path.join(self._root, self.processed_folder, self.multi_test_file))

    def __getitem__(self, index):
        if self._train:
            img, target_l, target_r = self.train_data[index], self.train_labels_l[index], self.train_labels_r[index]
        else:
            img, target_l, target_r = self.test_data[index], self.test_labels_l[index], self.test_labels_r[index]

        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        img = self._transform(img)

        return img, target_l, target_r

    def __len__(self):
        if self._train:
            return len(self.train_data)
        else:
            return len(self.test_data)
