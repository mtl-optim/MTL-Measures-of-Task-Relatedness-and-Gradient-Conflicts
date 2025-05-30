import sys
sys.path.append('../')
import random
from allennlp.modules.elmo import Elmo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from utils_data_load import *
from training_utils import *
import pickle
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=10)

args = parser.parse_args()

print(f"Method: {args.method}")
print(f"Epochs: {args.epochs}")
print(f"Seed: {args.seed}")

method = args.method
num_epochs = args.epochs
seed = args.seed

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

with open("balanced_datasets_2.pkl", "rb") as file:
    loaded_datasets = pickle.load(file)

train_msr = loaded_datasets["train_msr"]
train_rte = loaded_datasets["train_rte"]
train_qnli = loaded_datasets["train_qnli"]
train_qqp = loaded_datasets["train_qqp"]
train_mnli = loaded_datasets["train_mnli"]
train_sst = loaded_datasets["train_sst"]
train_cola = loaded_datasets["train_cola"]

print("Datasets Loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

glove_filepath = 'glove.840B.300d.txt'
embedding_dim = 300
glove_embeddings = load_glove_embeddings(glove_filepath, embedding_dim, device)

elmo_filepath = 'elmo_options.json'
elmo_weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
elmo = Elmo(elmo_filepath, elmo_weight_file, num_output_representations=1, dropout=0).to(device)

tokenizer = word_tokenize
cola_train_dataset = GLUEDataset(train_cola, tokenizer, glove_embeddings, elmo, max_length=150)
sst_train_dataset = GLUEDataset(train_sst, tokenizer, glove_embeddings, elmo, max_length=150)
mnli_train_dataset = GLUEDatasetTwoSentence(train_mnli, tokenizer, glove_embeddings, elmo, max_length=150)
msr_train_dataset = GLUEDatasetTwoSentence(train_msr, tokenizer, glove_embeddings, elmo, max_length=150)
qnli_train_dataset = GLUEDatasetTwoSentence(train_qnli, tokenizer, glove_embeddings, elmo, max_length=150)
qqp_train_dataset = GLUEDatasetTwoSentence(train_qqp, tokenizer, glove_embeddings, elmo, max_length=150)
rte_train_dataset = GLUEDatasetTwoSentence(train_rte, tokenizer, glove_embeddings, elmo, max_length=150)

batch_size = 8
cola_train_loader = DataLoader(cola_train_dataset, batch_size=batch_size )
sst_train_loader = DataLoader(sst_train_dataset, batch_size=batch_size )
mnli_train_loader = DataLoader(mnli_train_dataset, batch_size=batch_size )
msr_train_loader = DataLoader(msr_train_dataset, batch_size=batch_size )
qnli_train_loader = DataLoader(qnli_train_dataset, batch_size=batch_size )
qqp_train_loader = DataLoader(qqp_train_dataset, batch_size=batch_size )
rte_train_loader = DataLoader(rte_train_dataset, batch_size=batch_size )

embedding_dim = 300 + 1024  # GloVe (300) + ELMo (1024)
hidden_size = 1500

model = MultiTaskBiLSTM(embedding_dim, hidden_size, num_classes, task_types).to(device)
classification_loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

tasks_to_train = ["CoLA", "SST-2", "MNLI", "QNLI", "RTE", "QQP", "MRPC"]

tasks = {
    "CoLA": {"train_loader": cola_train_loader, "metric": "matthews"},
    "SST-2": {"train_loader": sst_train_loader, "metric": "accuracy"},
    "MNLI": {"train_loader": mnli_train_loader, "metric": "accuracy"},
    "QNLI": {"train_loader": qnli_train_loader, "metric": "accuracy"},
    "RTE": {"train_loader": rte_train_loader, "metric": "accuracy"},
    "QQP": {"train_loader": qqp_train_loader, "metric": "f1_accuracy"},
    "MRPC": {"train_loader": msr_train_loader, "metric": "f1_accuracy"},
}

shared_bilstm_params = model.bilstm.parameters()
shared_bilstm_pair_params = model.bilstm_pair.parameters()

all_task_params = []
for task_name in model.classifiers:
    all_task_params.extend(list(model.classifiers[task_name].parameters()))

# shared_bilstm_optimizer = torch.optim.Adam(shared_bilstm_params, lr=1e-4)
# shared_bilstm_pair_optimizer = torch.optim.Adam(shared_bilstm_pair_params, lr=1e-4)
# task_optimizer = torch.optim.Adam(all_task_params, lr=1e-4)

grad_bilstm_dims = []
for param in model.bilstm.parameters():
    grad_bilstm_dims.append(param.data.numel())
grads_bilstm = torch.Tensor(sum(grad_bilstm_dims), 7).to(device)

grad_bilstm_pair_dims = []
for param in model.bilstm_pair.parameters():
    grad_bilstm_pair_dims.append(param.data.numel())
grads_bilstm_pair = torch.Tensor(sum(grad_bilstm_pair_dims), 5).to(device)

rng = np.random.default_rng()

def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for p in m.parameters():
        grad = p.grad
        if grad is not None:
            grad_cur = grad.data.detach().clone()
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, task].copy_(grad_cur.data.view(-1))
        cnt += 1
        
def overwrite_grad(m, newgrad, grad_dims, num_grads):
    newgrad = newgrad * num_grads # to match the sum loss
    cnt = 0
    for param in m.parameters():
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = newgrad[beg: en].contiguous().view(param.data.size())
        param.grad = this_grad.data.clone().to(device)
        cnt += 1

# def zero_grad_shared_modules(model):
#     shared_modules = [model.bilstm, model.bilstm_pair]
#     for module in shared_modules:
#         for param in module.parameters():
#             if param.grad is not None:
#                 param.grad.zero_()

def custom_backward(method, bilstm, bilstm_pair, task_types, losses):
    task_types = ['MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE', 'CoLA', 'SST-2']
    pair_task_types = ['MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE']
    for task, loss in losses.items():
        i = task_types.index(task)
        loss.backward(retain_graph=True)
        if task in pair_task_types:
            grad2vec(bilstm_pair, grads_bilstm_pair, grad_bilstm_pair_dims, i)
        grad2vec(bilstm, grads_bilstm, grad_bilstm_dims, i)
        for module in [bilstm,bilstm_pair]:
            for param in module.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        shared_modules = [model.bilstm, model.bilstm_pair]
        for module in shared_modules:
            for param in module.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        # shared_bilstm_optimizer.zero_grad()
        # shared_bilstm_pair_optimizer.zero_grad()
    if method == "classic":
        g_pair = gd(grads_bilstm_pair)
        g = gd(grads_bilstm)
    elif method == "pcgrad":
        g_pair = pcgrad(grads_bilstm_pair, rng, 5)
        g = pcgrad(grads_bilstm, rng, 7)
    elif method == "cagrad":
        g_pair = cagrad(grads_bilstm_pair, num_tasks=5)
        g = cagrad(grads_bilstm,num_tasks=7)
    elif method == "graddrop":
        g_pair = graddrop(grads_bilstm_pair)
        g = graddrop(grads_bilstm)
    elif method == "mgd":
        g_pair = mgd(grads_bilstm_pair)
        g = mgd(grads_bilstm)
    elif method == "modifiedgd":
        g_pair = modified_mgd5(grads_bilstm_pair)
        g = modified_mgd7(grads_bilstm)
    overwrite_grad(bilstm_pair, g_pair, grad_bilstm_pair_dims, num_grads=5)
    overwrite_grad(bilstm, g, grad_bilstm_dims, num_grads = 7)

for epoch in range(num_epochs):
    model.train()
    task_iters = {task: iter(loader["train_loader"]) for task, loader in tasks.items()}
    for step in tqdm(range(min(len(loader["train_loader"]) for loader in tasks.values())), desc=f"Epoch {epoch+1}"):
        losses = {}
        total_loss = 0
        for task_name, loader in tasks.items():
            train_dataset_ = iter(task_iters[task_name])
            try:
                batch = next(train_dataset_)
                embeddings1 = batch['embeddings1'].to(device)
                if task_types[task_name] is "single":
                    labels = batch['label'].to(device)
                    outputs = model(task_name, embeddings1)
                else:
                    embeddings2 = batch['embeddings2'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(task_name, embeddings1, embeddings2)
                    
                if task_name in ['CoLA', "SST-2", 'MRPC', "QQP", "MNLI", "QNLI", 'RTE', 'WNLI']:  # Classification tasks
                    loss = classification_loss(outputs, labels)
                else:
                    raise ValueError(f"Unknown task type for {task_name}")
                if math.isnan(loss.item()):
                    print(f"NaN loss at step {step} for task {task_name}: {loss}")
            except StopIteration:
                print("StopIteration Exception")
                continue
            losses[task_name] = loss
        
        if method == "classic":
            optimizer.zero_grad()
            loss_ = sum(list(losses.values()))
            loss_.backward()
        else:
            optimizer.zero_grad()
            custom_backward(method, model.bilstm, model.bilstm_pair, task_types, losses)
        # shared_bilstm_optimizer.step()
        # shared_bilstm_pair_optimizer.step()
        # task_optimizer.step()
        optimizer.step()
    
    print(f"Epoch {epoch+1}")
    print(losses)



torch.save(model.state_dict(), f'./models/glue_{method}_{num_epochs}ep_{seed}seed.pth')



