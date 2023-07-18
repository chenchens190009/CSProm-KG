import os
import random
from tqdm import tqdm
from collections import defaultdict as ddict
import numpy as np
import pandas as pd
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.graph_models import ConvE, RotatE, TransE, DistMult, Null

GRAPH_MODEL_CLASS = {
    'conve': ConvE,
    'rotate': RotatE,
    'transe': TransE,
    'distmult': DistMult,
    'null': Null,
}


def get_num(dataset_path, dataset, mode='entity'):  # mode: {entity, relation}
    return int(open(os.path.join(dataset_path, dataset, mode + '2id.txt'), encoding='utf-8').readline().strip())


def read(configs, dataset_path, dataset, filename):
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name) as file:
        lines = file.read().strip().split('\n')
    n_triples = int(lines[0])
    triples = []
    for line in lines[1:]:
        split = line.split(' ')
        for i in range(3):
            split[i] = int(split[i])
        triples.append(split)
    assert n_triples == len(triples), 'number of triplets is not correct.'
    return triples


def read_file(configs, dataset_path, dataset, filename, mode='desc'):
    id2name = []
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name, encoding='utf-8') as file:
        lines = file.read().strip('\n').split('\n')
    for i in range(1, len(lines)):
        ids, name = lines[i].split('\t')
        if mode == 'desc':
            name = name.split(' ')
            name = ' '.join(name)
        id2name.append(name)
    return id2name


def read_name(configs, dataset_path, dataset):
    ent_name_file = 'entityid2name.txt'
    rel_name_file = 'relationid2name.txt'
    ent_names = read_file(configs, dataset_path, dataset, ent_name_file, 'name')
    rel_names = read_file(configs, dataset_path, dataset, rel_name_file, 'name')
    return ent_names, rel_names


def get_gt(configs, triples):
    tail_gt, head_gt = ddict(list), ddict(list)
    for triple in triples:
        if not configs.is_temporal:
            head, tail, rel = triple
            tail_gt[(head, rel)].append(tail)
            head_gt[(tail, rel + configs.n_rel)].append(head)
        else:
            head, tail, rel, timestamp = triple
            tail_gt[(head, rel, timestamp)].append(tail)
            head_gt[(tail, rel + configs.n_rel, timestamp)].append(head)
    return tail_gt, head_gt


def dataloader_output_to_tensor(output_dict, key, padding_value=None, return_list=False):
    tensor_out = [out[key] for out in output_dict]
    if return_list:
        return tensor_out
    if not isinstance(tensor_out[0], torch.LongTensor) and not isinstance(tensor_out[0], torch.FloatTensor):
        tensor_out = [torch.LongTensor(value) for value in tensor_out]
    if padding_value is None:
        tensor_out = torch.stack(tensor_out, dim=0)
    else:
        tensor_out = pad_sequence(tensor_out, batch_first=True, padding_value=padding_value)
    return tensor_out


def _get_performance(ranks):
    ranks = np.array(ranks, dtype=np.float)
    out = dict()
    out['mr'] = ranks.mean(axis=0)
    out['mrr'] = (1. / ranks).mean(axis=0)
    out['hit1'] = np.sum(ranks == 1, axis=0) / len(ranks)
    out['hit3'] = np.sum(ranks <= 3, axis=0) / len(ranks)
    out['hit10'] = np.sum(ranks <= 10, axis=0) / len(ranks)
    return out


def get_performance(model, tail_ranks, head_ranks):
    tail_out = _get_performance(tail_ranks)
    head_out = _get_performance(head_ranks)
    mr = np.array([tail_out['mr'], head_out['mr']])
    mrr = np.array([tail_out['mrr'], head_out['mrr']])
    hit1 = np.array([tail_out['hit1'], head_out['hit1']])
    hit3 = np.array([tail_out['hit3'], head_out['hit3']])
    hit10 = np.array([tail_out['hit10'], head_out['hit10']])

    val_mrr = mrr.mean().item()
    model.log('val_mrr', val_mrr)
    perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    perf = pd.DataFrame(perf, index=['tail ranking', 'head ranking'])
    perf.loc['mean ranking'] = perf.mean(axis=0)
    for hit in ['hit@1', 'hit@3', 'hit@5', 'hit@10']:
        if hit in list(perf.columns):
            perf[hit] = perf[hit].apply(lambda x: '%.2f%%' % (x * 100))
    return perf


# Helper functions from fastai
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


# Implementation from fastai https://github.com/fastai/fastai2/blob/master/fastai2/layers.py#L338
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, e: float = 0.1, reduction='mean'):
        super().__init__()
        self.e, self.reduction = e, reduction

    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # (1-e)* H(q,p) + e*H(u,p)
        return (1 - self.e) * nll + self.e * (loss / c)


def get_loss_fn(configs):
    if configs.label_smoothing == 0:
        return torch.nn.CrossEntropyLoss()
    elif configs.label_smoothing != 0:
        return LabelSmoothingCrossEntropy(configs.label_smoothing)


def get_lar_sample_bank(configs, text_dict):
    def process(tokens, stop_words):
        tokens = map(lambda x: x.lower(), tokens)
        tokens = filter(lambda x: x not in stop_words, tokens)
        return list(tokens)

    ent_names = text_dict['ent_names']
    ent_descs = text_dict['ent_descs']
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
    name_token2ids, desc_token2ids = ddict(list), ddict(list)
    name_id2tokens, desc_id2tokens = dict(), dict()
    desc_stop_words = stop_words

    for i in tqdm(range(configs.n_ent), desc='Processing'):
        name, desc = ent_names[i], ent_descs[i]
        if configs.dataset == 'WN18RR':
            name = name.split(' , ')[0]
        name_tokens, desc_tokens = tokenizer.tokenize(name), tokenizer.tokenize(desc)
        name_tokens, desc_tokens = process(name_tokens, stop_words), process(desc_tokens, desc_stop_words)
        for token in name_tokens:
            name_token2ids[token].append(i)
        name_id2tokens[i] = list(set(name_tokens))
        for token in desc_tokens:
            desc_token2ids[token].append(i)
        desc_id2tokens[i] = list(set(desc_tokens))
    if configs.max_lar_samples > 0:
        for key, value in name_token2ids.items():
            if len(value) > configs.max_lar_samples:
                name_token2ids[key] = random.sample(value, configs.max_lar_samples)
        for key, value in desc_token2ids.items():
            if len(value) > configs.max_lar_samples:
                desc_token2ids[key] = random.sample(value, configs.max_lar_samples)
    name_lars, desc_lars = {}, {}
    for i in tqdm(range(configs.n_ent), desc='Processing name'):
        lar_list = list(set([ids for token in name_id2tokens[i] for ids in name_token2ids[token]]))
        if configs.max_lar_samples > 0 and len(lar_list) > configs.max_lar_samples:
            lar_list = random.sample(lar_list, configs.max_lar_samples)
        name_lars[i] = lar_list
    for i in tqdm(range(configs.n_ent), desc='Processing desc'):
        lar_list = list(set([ids for token in desc_id2tokens[i] for ids in desc_token2ids[token]]))
        if configs.max_lar_samples > 0 and len(lar_list) > configs.max_lar_samples:
            lar_list = random.sample(lar_list, configs.max_lar_samples)
        desc_lars[i] = lar_list

    lars_dict = {'name_lars': name_lars, 'desc_lars': desc_lars}
    # return token_dict, negs_dict
    return lars_dict
