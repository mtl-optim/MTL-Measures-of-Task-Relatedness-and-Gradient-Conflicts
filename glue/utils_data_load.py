import numpy as np
import torch
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import Dataset
import re
import copy
from copy import deepcopy
from min_norm_solvers import MinNormSolver
from scipy.optimize import minimize, Bounds, minimize_scalar
import itertools

def load_glove_embeddings(filepath, embedding_dim, device="cuda"):
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            if len(values) != embedding_dim + 1:
                continue
            word = values[0]
            embedding = torch.tensor(np.asarray(values[1:], dtype='float32'), device=device)  # Store on CUDA
            embeddings_index[word] = embedding
    return embeddings_index

def check_string_compatibility(text):
    try:
        re.sub(r'\s+', ' ', text)
        return True
    except TypeError:
        return False
    
def load_cola_dataset():
    train_cola = pd.read_table('./cola/train.tsv', header=None, names=["source", "label", "unused", "sentence"])
    dev_cola = pd.read_table('./cola/dev.tsv', header=None, names=["source", "label", "unused", "sentence"])
    train_cola['sentence'] = train_cola['sentence'].astype(str)
    dev_cola['sentence'] = dev_cola['sentence'].astype(str)
    return train_cola, dev_cola
    
def load_sst_dataset():
    train_sst = pd.read_table('./sst/train.tsv')
    dev_sst = pd.read_table('./sst/dev.tsv')
    train_sst['sentence'] = train_sst['sentence'].astype(str)
    dev_sst['sentence'] = dev_sst['sentence'].astype(str)
    return train_sst, dev_sst

def load_mnli_dataset():
    train_mnli = pd.read_table('./mnli/train.tsv',sep = '\t', skiprows=1, error_bad_lines=False, header=None, names=['index','genre','filename','year','old_index','source1','source2','sentence1','sentence2','label'])
    dev1_mnli = pd.read_table('./mnli/dev_matched.tsv',sep = '\t', skiprows=1, error_bad_lines=False, header=None, names=['index','genre','filename','year','old_index','source1','source2','sentence1','sentence2','label'])
    dev2_mnli = pd.read_table('./mnli/dev_mismatched.tsv',sep = '\t', skiprows=1, error_bad_lines=False, header=None, names=['index','genre','filename','year','old_index','source1','source2','sentence1','sentence2','label'])
    dev_mnli = pd.concat([dev1_mnli, dev2_mnli], ignore_index=True)
    train_mnli = train_mnli.dropna(subset=['label'])
    dev_mnli = dev_mnli.dropna(subset=['label'])
    train_mnli['label'] = train_mnli['label'].map({'neutral': int(0), 'contradiction': int(1), 'entailment': int(2)})
    dev_mnli['label'] = dev_mnli['label'].map({'neutral': int(0), 'contradiction': int(1), 'entailment': int(2)})
    train_mnli['is_valid_string'] = train_mnli['sentence1'].apply(check_string_compatibility)
    train_mnli = train_mnli[train_mnli['is_valid_string']].drop(columns=['is_valid_string'])
    train_mnli['sentence1'] = train_mnli['sentence1'].astype(str)
    train_mnli['sentence2'] = train_mnli['sentence2'].astype(str)
    dev_mnli['sentence1'] = dev_mnli['sentence1'].astype(str)
    dev_mnli['sentence2'] = dev_mnli['sentence2'].astype(str)
    return train_mnli, dev_mnli

def load_sts_dataset():
    train_sts = pd.read_table('./sts/train.tsv',sep = '\t', error_bad_lines=False)
    dev_sts = pd.read_table('./sts/dev.tsv',sep = '\t', error_bad_lines=False)
    train_sts = train_sts.dropna(subset=['sentence1'])
    train_sts = train_sts.dropna(subset=['sentence2'])
    train_sts = train_sts.dropna(subset=['score'])
    dev_sts = dev_sts.dropna(subset=['sentence1'])
    dev_sts = dev_sts.dropna(subset=['sentence2'])
    dev_sts = dev_sts.dropna(subset=['score'])
    train_sts['sentence1'] = train_sts['sentence1'].astype(str)
    train_sts['sentence2'] = train_sts['sentence2'].astype(str)
    dev_sts['sentence1'] = dev_sts['sentence1'].astype(str)
    dev_sts['sentence2'] = dev_sts['sentence2'].astype(str)
    train_sts['score'] = train_sts['score']/5.0
    dev_sts['score'] = dev_sts['score']/5.0
    return train_sts, dev_sts

def load_qqp_dataset():
    train_qqp = pd.read_table('./qqp/train.tsv',sep = '\t', skiprows=1, error_bad_lines=False, names=['id','qid1','qid2','sentence1','sentence2','label'])
    dev_qqp = pd.read_table('./qqp/dev.tsv',sep = '\t', skiprows=1, error_bad_lines=False, names=['id','qid1','qid2','sentence1','sentence2','label'])
    test_qqp = pd.read_table('./qqp/test.tsv',sep = '\t', error_bad_lines=False, quoting=csv.QUOTE_NONE)
    train_qqp['sentence1'] = train_qqp['sentence1'].astype(str)
    train_qqp['sentence2'] = train_qqp['sentence2'].astype(str)
    dev_qqp['sentence1'] = dev_qqp['sentence1'].astype(str)
    dev_qqp['sentence2'] = dev_qqp['sentence2'].astype(str)
    return train_qqp, dev_qqp

def load_qnli_dataset():
    train_qnli = pd.read_table('./qnli/train.tsv', skiprows=1,sep = '\t', error_bad_lines=False, names=['index','sentence1','sentence2','label'])
    train_qnli['label'] = train_qnli['label'].map({'not_entailment': int(0), 'entailment': int(1)})
    dev_qnli = pd.read_table('./qnli/dev.tsv', skiprows=1,sep = '\t', error_bad_lines=False, names=['index','sentence1','sentence2','label'])
    dev_qnli['label'] = dev_qnli['label'].map({'not_entailment': 0, 'entailment': 1})
    test_qnli = pd.read_table('./qnli/test.tsv',sep = '\t', error_bad_lines=False, quoting=csv.QUOTE_NONE)
    train_qnli = train_qnli.dropna(subset=['label'])
    dev_qnli = dev_qnli.dropna(subset=['label'])
    train_qnli['label'] = train_qnli['label'].astype(int)
    dev_qnli['label'] = dev_qnli['label'].astype(int)
    train_qnli['sentence1'] = train_qnli['sentence1'].astype(str)
    train_qnli['sentence2'] = train_qnli['sentence2'].astype(str)
    dev_qnli['sentence1'] = dev_qnli['sentence1'].astype(str)
    dev_qnli['sentence2'] = dev_qnli['sentence2'].astype(str)
    return train_qnli, dev_qnli

def load_rte_dataset():
    train_rte = pd.read_table('./rte/train.tsv',sep = '\t', error_bad_lines=False)
    train_rte['label'] = train_rte['label'].map({'not_entailment': int(0), 'entailment': int(1)})
    dev_rte = pd.read_table('./rte/dev.tsv',sep = '\t', error_bad_lines=False)
    dev_rte['label'] = dev_rte['label'].map({'not_entailment': 0, 'entailment': 1})
    test_rte = pd.read_table('./rte/test.tsv',sep = '\t', error_bad_lines=False, quoting=csv.QUOTE_NONE)
    train_rte = train_rte.dropna(subset=['label'])
    dev_rte = dev_rte.dropna(subset=['label'])
    train_rte['label'] = train_rte['label'].astype(int)
    dev_rte['label'] = dev_rte['label'].astype(int)
    train_rte['sentence1'] = train_rte['sentence1'].astype(str)
    train_rte['sentence2'] = train_rte['sentence2'].astype(str)
    dev_rte['sentence1'] = dev_rte['sentence1'].astype(str)
    dev_rte['sentence2'] = dev_rte['sentence2'].astype(str)
    return train_rte, dev_rte

def load_wnli_dataset():
    train_wnli = pd.read_table('./wnli/train.tsv',sep = '\t', error_bad_lines=False)
    dev_wnli = pd.read_table('./wnli/dev.tsv',sep = '\t', error_bad_lines=False)
    train_wnli['sentence1'] = train_wnli['sentence1'].astype(str)
    train_wnli['sentence2'] = train_wnli['sentence2'].astype(str)
    dev_wnli['sentence1'] = dev_wnli['sentence1'].astype(str)
    dev_wnli['sentence2'] = dev_wnli['sentence2'].astype(str)
    return train_wnli, dev_wnli

def load_mrpc_dataset():
    train_msr = pd.read_csv('./msr/msr_paraphrase_train.txt', skiprows=1, sep='\t', quoting=3, names=['label','id1','id2','sentence1','sentence2'])
    train_msr, dev_msr = train_test_split(train_msr, test_size=0.2, random_state=42)
    train_msr['sentence1'] = train_msr['sentence1'].astype(str)
    train_msr['sentence2'] = train_msr['sentence2'].astype(str)
    dev_msr['sentence1'] = dev_msr['sentence1'].astype(str)
    dev_msr['sentence2'] = dev_msr['sentence2'].astype(str)
    return train_msr, dev_msr

def preprocess_sentence_with_embeddings(sentence, tokenizer, glove_embeddings, elmo, device="cuda"):
    tokens = tokenizer(sentence)
    glove_vectors = []
    for token in tokens:
        if token in glove_embeddings:
            glove_vectors.append(glove_embeddings[token])  # Already on CUDA
        else:
            glove_vectors.append(torch.zeros(300, device=device))  # Also on CUDA
    
    character_ids = batch_to_ids([tokens]).to(device)  # Move character IDs to CUDA
    elmo_output = elmo(character_ids)
    elmo_vectors = elmo_output['elmo_representations'][0].detach()[0]  # Already on CUDA

    glove_vectors = torch.stack(glove_vectors)  # Convert list of tensors to a single tensor
    concatenated_embeddings = torch.cat([glove_vectors, elmo_vectors], dim=1)  # Ensure same dtype & device

    return concatenated_embeddings

class GLUEDataset(Dataset):
    def __init__(self, dataframe, tokenizer, glove_embeddings, elmo, device="cuda", max_length=50):
        self.sentences = dataframe['sentence'].values
        self.labels = dataframe['label'].values
        self.tokenizer = tokenizer
        self.glove_embeddings = glove_embeddings
        self.elmo = elmo
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        embeddings = preprocess_sentence_with_embeddings(sentence, self.tokenizer, self.glove_embeddings, self.elmo, self.device)

        if embeddings.shape[0] > self.max_length:
            embeddings = embeddings[:self.max_length]
        else:
            pad_length = self.max_length - embeddings.shape[0]
            pad_vector = torch.zeros((pad_length, embeddings.shape[1]), dtype=torch.float, device=self.device)
            embeddings = torch.cat([embeddings, pad_vector], dim=0)

        return {
            'embeddings1': embeddings,  # Already on CUDA
            'label': torch.tensor(label, dtype=torch.long, device=self.device)  # Move label to CUDA
        }
        
class GLUEDatasetTwoSentence(Dataset):
    def __init__(self, dataframe, tokenizer, glove_embeddings, elmo, device="cuda", max_length=50):
        self.sent1 = dataframe['sentence1'].values
        self.sent2 = dataframe['sentence2'].values
        self.labels = dataframe['label'].values
        self.tokenizer = tokenizer
        self.glove_embeddings = glove_embeddings
        self.elmo = elmo
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.sent1)

    def __getitem__(self, idx):
        sentence1 = self.sent1[idx]
        sentence2 = self.sent2[idx]
        label = self.labels[idx]

        embeddings1 = preprocess_sentence_with_embeddings(sentence1, self.tokenizer, self.glove_embeddings, self.elmo, self.device)
        embeddings2 = preprocess_sentence_with_embeddings(sentence2, self.tokenizer, self.glove_embeddings, self.elmo, self.device)

        embeddings1 = self._pad_or_truncate(embeddings1)
        embeddings2 = self._pad_or_truncate(embeddings2)

        return {
            'embeddings1': embeddings1,
            'embeddings2': embeddings2,
            'label': torch.tensor(label, dtype=torch.long, device=self.device)
        }

    def _pad_or_truncate(self, embeddings):
        if embeddings.shape[0] > self.max_length:
            return embeddings[:self.max_length]
        else:
            pad_length = self.max_length - embeddings.shape[0]
            pad_vector = torch.zeros((pad_length, embeddings.shape[1]), dtype=torch.float, device=self.device)
            embeddings = torch.cat([embeddings, pad_vector], dim=0)
            return embeddings

class GLUEDatasetTest(Dataset):
    def __init__(self, dataframe, tokenizer, glove_embeddings, elmo, device="cuda", max_length=50):
        self.sentences = dataframe['sentence'].values
        # self.labels = dataframe['label'].values
        self.tokenizer = tokenizer
        self.glove_embeddings = glove_embeddings
        self.elmo = elmo
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # label = self.labels[idx]
        embeddings = preprocess_sentence_with_embeddings(sentence, self.tokenizer, self.glove_embeddings, self.elmo, self.device)

        if embeddings.shape[0] > self.max_length:
            embeddings = embeddings[:self.max_length]
        else:
            pad_length = self.max_length - embeddings.shape[0]
            pad_vector = torch.zeros((pad_length, embeddings.shape[1]), dtype=torch.float, device=self.device)
            embeddings = torch.cat([embeddings, pad_vector], dim=0)
        return {
            'embeddings1': embeddings,
            # 'label': torch.tensor(label, dtype=torch.long)
        }
        
class GLUEDatasetTwoSentenceTest(Dataset):
    def __init__(self, dataframe, tokenizer, glove_embeddings, elmo, device="cuda", max_length=50):
        self.sent1 = dataframe['sentence1'].values
        self.sent2 = dataframe['sentence2'].values
        # self.labels = dataframe['label'].values
        self.tokenizer = tokenizer
        self.glove_embeddings = glove_embeddings
        self.elmo = elmo
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.sent1)

    def __getitem__(self, idx):
        sentence1 = self.sent1[idx]
        sentence2 = self.sent2[idx]
        # label = self.labels[idx]

        embeddings1 = preprocess_sentence_with_embeddings(sentence1, self.tokenizer, self.glove_embeddings, self.elmo, self.device)
        embeddings2 = preprocess_sentence_with_embeddings(sentence2, self.tokenizer, self.glove_embeddings, self.elmo, self.device)

        embeddings1 = self._pad_or_truncate(embeddings1)
        embeddings2 = self._pad_or_truncate(embeddings2)

        return {
            'embeddings1': embeddings1,
            'embeddings2': embeddings2,
            # 'label': torch.tensor(label, dtype=torch.long, device=self.device)
        }

    def _pad_or_truncate(self, embeddings):
        if embeddings.shape[0] > self.max_length:
            return embeddings[:self.max_length]
        else:
            pad_length = self.max_length - embeddings.shape[0]
            pad_vector = torch.zeros((pad_length, embeddings.shape[1]), dtype=torch.float, device=self.device)
            embeddings = torch.cat([embeddings, pad_vector], dim=0)
            return embeddings