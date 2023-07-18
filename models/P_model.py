import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoConfig
from helper import get_performance, get_loss_fn, GRAPH_MODEL_CLASS
from models.prompter import Prompter
from models.bert_for_layerwise import BertModelForLayerwise


class KGCPromptTuner(pl.LightningModule):
    def __init__(self, configs, text_dict, gt):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']

        self.ent_embed = nn.Embedding(self.configs.n_ent, self.configs.embed_dim)
        if self.configs.graph_model in ['transe', 'rotate']:
            self.rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            self.rel_embed = nn.Embedding(self.configs.n_rel * 2, self.configs.embed_dim)

        self.plm_configs = AutoConfig.from_pretrained(configs.pretrained_model)
        self.plm_configs.prompt_length = self.configs.prompt_length
        self.plm_configs.prompt_hidden_dim = self.configs.prompt_hidden_dim
        self.plm = BertModelForLayerwise.from_pretrained(configs.pretrained_model)

        self.prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)
        self.fc = nn.Linear(configs.prompt_length * self.plm_configs.hidden_size, configs.embed_dim)
        if configs.prompt_length > 0:
            for p in self.plm.parameters():
                p.requires_grad = False

        self.graph_model = GRAPH_MODEL_CLASS[self.configs.graph_model](configs)
        if configs.n_lar > 0:
            self.lar_loss_fn = nn.TripletMarginWithDistanceLoss(
                margin=configs.gamma,
                distance_function=lambda x, y: self.graph_model.score_fn(x, y[0], y[1]),
            )

        self.history = {'perf': ..., 'loss': []}
        self.loss_fn = get_loss_fn(configs)
        self._MASKING_VALUE = -1e4 if self.configs.use_fp16 else -1e9
        if self.configs.alpha_step > 0:
            self.alpha = 0.
        else:
            self.alpha = self.configs.alpha

    def forward(self, ent_rel, src_ids, src_mask):
        bs = ent_rel.size(0)
        all_ent_embed = self.ent_embed.weight
        if self.configs.graph_model in ['transe', 'rotate']:
            all_rel_embed = torch.cat([self.rel_embed.weight, -self.rel_embed.weight], dim=0)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            all_rel_embed = self.rel_embed.weight

        ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        ent_embed = all_ent_embed[ent]
        rel_embed = all_rel_embed[rel]
        prompt = self.prompter(torch.stack([ent_embed, rel_embed], dim=1))
        prompt_attention_mask = torch.ones(ent_embed.size(0), self.configs.prompt_length * 2).type_as(src_mask)
        src_mask = torch.cat((prompt_attention_mask, src_mask), dim=1)
        output = self.plm(input_ids=src_ids, attention_mask=src_mask, layerwise_prompt=prompt)

        # last_hidden_state -- .shape: (batch_size, seq_len, model_dim)
        last_hidden_state = output.last_hidden_state

        ent_rel_state = last_hidden_state[:, :self.configs.prompt_length * 2]
        plm_ent_embed, plm_rel_embed = torch.chunk(ent_rel_state, chunks=2, dim=1)
        plm_ent_embed = self.fc(plm_ent_embed.reshape(ent_embed.size(0), -1))
        plm_rel_embed = self.fc(plm_rel_embed.reshape(rel_embed.size(0), -1))

        # pred -- .shape: (batch_size, embed_dim)
        pred = self.graph_model(plm_ent_embed, plm_rel_embed)
        # logits -- .shape: (batch_size, n_ent)
        logits = self.graph_model.get_logits(pred, all_ent_embed)
        return logits, pred

    def training_step(self, batched_data, batch_idx):
        if self.configs.alpha_step > 0 and self.alpha < self.configs.alpha:
            self.alpha = min(self.alpha + self.configs.alpha_step, self.configs.alpha)
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # ent_rel .shape: (batch_size, 2)
        ent_rel = batched_data['ent_rel']
        tgt_ent = batched_data['tgt_ent']
        labels = batched_data['labels']
        lars = batched_data['lars'] if self.configs.n_lar > 0 else None

        logits, pred = self(ent_rel, src_ids, src_mask)
        loss = self.loss_fn(logits, labels)
        if self.configs.n_lar > 0:
            lar_ent_embed = self.ent_embed
            # pos, neg -- .shape: (batch_size, 1, embed_dim), pos_bias, neg_bias -- .shape: (batch_size, 1)
            pos, lar = lar_ent_embed(labels).unsqueeze(1), torch.mean(lar_ent_embed(lars), dim=1, keepdim=True)
            pos_bias, lar_bias = self.graph_model.bias[labels].unsqueeze(-1), torch.mean(self.graph_model.bias[lars], dim=-1, keepdim=True)
            lar_loss = self.lar_loss_fn(anchor=pred, positive=(pos, pos_bias), negative=(lar, lar_bias))
            loss = loss + self.alpha * lar_loss

        self.history['loss'].append(loss.detach().item())
        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx, dataset_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # test_triples .shape: (batch_size, 3)
        test_triples = batched_data['triple']
        # ent_rel .shape: (batch_size, 2)
        ent_rel = batched_data['ent_rel']
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        # tgt_ent -- .type: list
        tgt_ent = batched_data['tgt_ent']
        gt = self.all_tail_gt if dataset_idx == 0 else self.all_head_gt
        logits, _ = self(ent_rel, src_ids, src_mask)
        logits = logits.detach()
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            if self.configs.is_temporal:
                tgt_filter = gt[(hi, ri, test_triples[i][3])]
            else:
                # tgt_filter .type: list()
                tgt_filter = gt[(hi, ri)]
            ## store target score
            tgt_score = logits[i, ti].item()
            ## remove the scores of the entities we don't care
            logits[i, tgt_filter] = self._MASKING_VALUE
            ## recover the target values
            logits[i, ti] = tgt_score
        _, argsort = torch.sort(logits, dim=1, descending=True)
        argsort = argsort.cpu().numpy()

        ranks = []
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            rank = np.where(argsort[i] == ti)[0][0] + 1
            ranks.append(rank)
        if self.configs.use_log_ranks:
            filename = os.path.join(self.configs.save_dir, f'Epoch-{self.current_epoch}-ranks.tmp')
            self.log_ranks(filename, test_triples, argsort, ranks, batch_idx)
        return ranks

    def validation_epoch_end(self, outs):
        tail_ranks = np.concatenate(outs[0])
        head_ranks = np.concatenate(outs[1])

        perf = get_performance(self, tail_ranks, head_ranks)
        print('Epoch:', self.current_epoch)
        print(perf)

    def test_step(self, batched_data, batch_idx, dataset_idx):
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.configs.lr)

    def log_ranks(self, filename, test_triples, argsort, ranks, batch_idx):
        assert len(test_triples) == len(ranks), 'length mismatch: test_triple, ranks!'
        with open(filename, 'a') as file:
            for i, triple in enumerate(test_triples):
                if not self.configs.is_temporal:
                    head, tail, rel = triple
                    timestamp = ''
                else:
                    head, tail, rel, timestamp = triple
                    timestamp = ' | ' + timestamp
                rank = ranks[i].item()
                triple_str = self.ent_names[head] + ' [' + self.ent_descs[head] + '] | ' + self.rel_names[rel]\
                    + ' | ' + self.ent_names[tail] + ' [' + self.ent_descs[tail] + '] ' + timestamp + '(%d %d %d)' % (head, tail, rel)
                file.write(str(batch_idx * self.configs.val_batch_size + i) + '. ' + triple_str + '=> ranks: ' + str(rank) + '\n')

                best10 = argsort[i, :10]
                for ii, ent in enumerate(best10):
                    ent = ent.item()
                    mark = '*' if (ii + 1) == rank else ' '
                    file.write('\t%2d%s ' % (ii + 1, mark) + self.ent_names[ent] + ' [' + self.ent_descs[ent] + ']' + ' (%d)' % ent + '\n')
