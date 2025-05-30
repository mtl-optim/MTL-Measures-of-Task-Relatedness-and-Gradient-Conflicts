import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
import copy
from min_norm_solvers import MinNormSolver
from scipy.optimize import minimize
import itertools

condition_numbers_pair = [[] for i in range(10)]
condition_numbers = [[] for i in range(21)]

task_types = {
    "CoLA": "single",
    "SST-2": "single",
    "MRPC": "pair",
    "QQP": "pair",
    "MNLI": "pair",
    "QNLI": "pair",
    "RTE": "pair",
}

num_classes = {
    "CoLA": 2,
    "SST-2": 2,
    "MRPC": 2,
    "QQP": 2,
    "MNLI": 3,
    "QNLI": 2,
    "RTE": 2,
}

class MultiTaskBiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_classes, task_types):

        super(MultiTaskBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.task_types = task_types  # Task types: single or pair
        self.dropout = 0.5
        
        # Shared BiLSTM for encoding
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Additional BiLSTM for sentence pairs
        self.bilstm_pair = nn.LSTM(
            input_size=hidden_size * 2 * 2,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Task-specific classifiers
        self.classifiers = nn.ModuleDict()
        for task, n_classes in num_classes.items():
            if task in task_types and task_types[task] == "pair":
                print(f"{task} classifier - T2")
                self.classifiers[task] = nn.Sequential(
                    nn.Dropout(p=self.dropout),
                    nn.Linear(hidden_size * 8, 512),  # [u; v; |u - v|; u * v]
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(512, n_classes)
                )
            else:  # Single-sentence tasks
                print(f"{task} classifier - T3")
                self.classifiers[task] = nn.Sequential(
                    nn.Dropout(p=self.dropout),
                    nn.Linear(hidden_size * 2, 512),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(512, n_classes)
                )
    
    def attention(self, lstm_out1, lstm_out2):
        # Compute attention scores
        attention_scores = torch.bmm(lstm_out1, lstm_out2.transpose(1, 2))  # (batch, seq1, seq2)
        
        # Normalize scores
        attention_weights1 = F.softmax(attention_scores, dim=-1)  # Weights for lstm_out2
        attention_weights2 = F.softmax(attention_scores.transpose(1, 2), dim=-1)  # Weights for lstm_out1
        
        # Compute context vectors
        context1 = torch.bmm(attention_weights1, lstm_out2)  # Context for lstm_out1
        context2 = torch.bmm(attention_weights2, lstm_out1)  # Context for lstm_out2
        
        return context1, context2
    
    def forward(self, task, embeddings1, embeddings2=None):
        if task not in self.task_types:
            raise ValueError(f"Unknown task: {task}")
        
        if self.task_types[task] == "single":
            # Single-sentence tasks
            lstm_out1, _ = self.bilstm(embeddings1)  # (batch, seq_len, hidden_size * 2)
            lstm_out1_pooled = torch.max(lstm_out1, dim=1)[0]  # Max pooling
            output = self.classifiers[task](lstm_out1_pooled)
        
        elif self.task_types[task] == "pair":
            # Two-sentence tasks
            if embeddings2 is None:
                raise ValueError(f"Embeddings2 must be provided for task {task}.")
            
            lstm_out1, _ = self.bilstm(embeddings1)  # (batch, seq_len1, hidden_size * 2)
            lstm_out2, _ = self.bilstm(embeddings2)  # (batch, seq_len2, hidden_size * 2)
            context1, context2 = self.attention(lstm_out1, lstm_out2)
            # Combine with original BiLSTM outputs
            lstm_out1_augmented = torch.cat([lstm_out1, context1], dim=-1)
            lstm_out2_augmented = torch.cat([lstm_out2, context2], dim=-1)
            
            # Encode contextualized representations
            lstm_out1_encoded, _ = self.bilstm_pair(lstm_out1_augmented)
            lstm_out2_encoded, _ = self.bilstm_pair(lstm_out2_augmented)
            u = torch.max(lstm_out1_encoded, dim=1)[0]
            v = torch.max(lstm_out2_encoded, dim=1)[0]
            features = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
            # print(features.shape)
            
            # Classify
            output = self.classifiers[task](features)
        
        return output
    
def calculate_metrics(task_name, predictions, labels):
    if task_name == "CoLA":
        return matthews_corrcoef(labels, predictions)
    elif task_name in ["SST-2", "WNLI", "RTE", "QNLI", "MNLI"]:
        return accuracy_score(labels, predictions)
    elif task_name in ["QQP", "MRPC"]:
        f1 = f1_score(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        return f1, acc
    elif task_name == "STS-B":
        pearson_corr = pearsonr(labels, predictions)[0]
        spearman_corr = spearmanr(labels, predictions)[0]
        return pearson_corr, spearman_corr
    else:
        raise ValueError(f"Unknown task {task_name}")
    
def compute_dynamic_threshold(cond_num, condition_numbers, percentile=50):
    if cond_num == -1:
        return True # singular matrix, not conflicting
    if len(condition_numbers) == 0:
        return True
    return cond_num < np.percentile(condition_numbers, percentile) # if >= then conflicting

def compute_condition_number(tensor):
    tensor = tensor.cpu()
    singular_values = torch.linalg.svdvals(tensor)
    condition_number = singular_values.max() / (singular_values.min() + 1e-6)
    return condition_number

def compute_condition_number_pair(g1, g2, eps=1e-10):
    pair = torch.stack([g1, g2])
    singular_values = np.linalg.svd(pair.cpu().numpy(), compute_uv=False)
    # print(singular_values)
    if singular_values.min() <= eps:
        # print("HERE INF")
        return -1
    return singular_values.max() / singular_values.min()

def there_are_conflicts(cond_num_matrix):
    for i in range(len(cond_num_matrix)):
        for j in range(len(cond_num_matrix[i])):
            if cond_num_matrix[i,j] > 0:
                return True
    return False

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

def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()

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
    modified_grad_vec = copy.deepcopy(grad_vec)
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

def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu() # [num_tasks, num_tasks]
    g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0,1) for x in x_start)
    cons=({'type':'eq','fun':lambda x:1-sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha*g0_norm+1e-8).item()
    def objfn(x):
        return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm+1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale== 0:
        return g
    elif rescale == 1:
        return g / (1+alpha**2)
    else:
        return g / (1 + alpha)
    
def modified_mgd5(grads, eps=1e-5):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g3 = grads[:,2]
    g4 = grads[:,3]
    g5 = grads[:,4]

    grad_vec = grads.t()

    norm_g1 = torch.norm(g1, 2)
    norm_g2 = torch.norm(g2, 2)
    norm_g3 = torch.norm(g3, 2)
    norm_g4 = torch.norm(g4, 2)
    norm_g5 = torch.norm(g5, 2)
    tensor_norms = [norm_g1, norm_g2, norm_g3, norm_g4, norm_g5]
    min_norm = min(tensor_norms)
    max_norm = max(tensor_norms)

    num_conflicts = 0
    idx_of_pair = 0

    for i, j in itertools.combinations(range(grads.shape[1]), 2):
        pair = torch.stack([grads[:, i], grads[:, j]], dim=0)
        U, singular_values, Vh = torch.linalg.svd(pair, full_matrices=False)
        if singular_values.min() <= eps:
            cond_num = -1
        else:
            cond_num = singular_values.max() / singular_values.min()
            condition_numbers_pair[idx_of_pair].append(cond_num.item())

        if len(condition_numbers_pair[idx_of_pair]) != 0 and cond_num != -1 and cond_num > np.percentile(condition_numbers_pair[idx_of_pair], 70):
            num_conflicts += 1

        idx_of_pair += 1

    if num_conflicts < 6:
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
    
def modified_mgd7(grads, eps=1e-5):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g3 = grads[:,2]
    g4 = grads[:,3]
    g5 = grads[:,4]
    g6 = grads[:,5]
    g7 = grads[:,6]

    grad_vec = grads.t()

    norm_g1 = torch.norm(g1, 2)
    norm_g2 = torch.norm(g2, 2)
    norm_g3 = torch.norm(g3, 2)
    norm_g4 = torch.norm(g4, 2)
    norm_g5 = torch.norm(g5, 2)
    norm_g6 = torch.norm(g6, 2)
    norm_g7 = torch.norm(g7, 2)
    tensor_norms = [norm_g1, norm_g2, norm_g3, norm_g4, norm_g5, norm_g6, norm_g7]
    min_norm = min(tensor_norms)
    max_norm = max(tensor_norms)

    num_conflicts = 0
    idx_of_pair = 0

    for i, j in itertools.combinations(range(grads.shape[1]), 2):
        pair = torch.stack([grads[:, i], grads[:, j]], dim=0)
        U, singular_values, Vh = torch.linalg.svd(pair, full_matrices=False)
        if singular_values.min() <= eps:
            cond_num = -1
        else:
            cond_num = singular_values.max() / singular_values.min()
            condition_numbers[idx_of_pair].append(cond_num.item())

        if len(condition_numbers[idx_of_pair]) != 0 and cond_num != -1 and cond_num >  np.percentile(condition_numbers[idx_of_pair], 70):
            num_conflicts += 1

        idx_of_pair += 1

    if num_conflicts < 13:
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