import torch
import torch.nn as nn
import torch.nn.functional as F


class Null(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.bias = nn.Parameter(torch.zeros(self.configs.n_ent))
        self.bias.requires_grad = False

    def forward(self, head_emb, rel_emb):
        return head_emb

    def get_logits(self, pred, ent_embed):
        logits = torch.mm(pred, ent_embed.transpose(1, 0))
        return logits

    def score_fn(self, pred, sample_embed, bias):
        # pred -- .shape: (batch_size, 1, embed_dim)
        pred = pred.unsqueeze(1)
        # scores -- .shape: (batch_size, n_samples)
        scores = torch.sum(pred * sample_embed, dim=-1)
        return scores


class TransE(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.bias = nn.Parameter(torch.zeros(self.configs.n_ent))
        self.bias.requires_grad = False

    def forward(self, head_emb, rel_emb):
        return head_emb + rel_emb

    # pred .shape: (batch_size, embed_dim), ent_embed .shape: (n_ent, embed_dim)
    def get_logits(self, pred, ent_embed):
        # logits .shape: (batch_size, n_ent, embed_dim)
        logits = pred.unsqueeze(1) - ent_embed
        # logits .shape: (batch_size, n_ent)
        logits = self.configs.loss_gamma - torch.norm(logits, p=1, dim=2)
        return logits

    # pred -- .shape: (batch_size, embed_dim)
    # sample_embed .shape: (batch_size, n_samples, embed_dim)
    def score_fn(self, pred, sample_embed, bias):
        scores = self.configs.loss_gamma - torch.norm(pred.unsqueeze(1) - sample_embed, p=1, dim=2)
        return scores


class DistMult(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.bias = nn.Parameter(torch.zeros(self.configs.n_ent))

    def forward(self, head_emb, rel_emb):
        return head_emb * rel_emb

    def get_logits(self, pred, ent_embed):
        logits = torch.mm(pred, ent_embed.transpose(1, 0))
        logits += self.bias.expand_as(logits)
        return logits

    # pred -- .shape: (batch_size, embed_dim)
    # sample_embed .shape: (batch_size, n_samples, embed_dim)
    def score_fn(self, pred, sample_embed, bias):
        # pred -- .shape: (batch_size, 1, embed_dim)
        pred = pred.unsqueeze(1)
        # scores -- .shape: (batch_size, n_samples)
        scores = torch.sum(pred * sample_embed, dim=-1) + bias
        return scores


class ConvE(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.configs.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.configs.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.configs.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.configs.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.configs.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.configs.num_filt, kernel_size=(self.configs.ker_sz, self.configs.ker_sz),
                                       stride=1, padding=0, bias=self.configs.bias)

        flat_sz_h = int(2 * self.configs.k_w) - self.configs.ker_sz + 1
        flat_sz_w = self.configs.k_h - self.configs.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.configs.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.configs.embed_dim)
        self.bias = nn.Parameter(torch.zeros(self.configs.n_ent))

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, e1_embed.size(-1))
        rel_embed = rel_embed.view(-1, 1, rel_embed.size(-1))
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.configs.k_w, self.configs.k_h))
        return stack_inp

    def forward(self, head_emb, rel_emb):
        stk_inp = self.concat(head_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # score = torch.sigmoid(x)
        return x

    def get_logits(self, pred, ent_embed):
        logits = torch.mm(pred, ent_embed.transpose(1, 0))
        logits += self.bias.expand_as(logits)
        return logits

    # pred -- .shape: (batch_size, embed_dim)
    # sample_embed .shape: (batch_size, n_samples, embed_dim)
    def score_fn(self, pred, sample_embed, bias):
        # pred -- .shape: (batch_size, 1, embed_dim)
        pred = pred.unsqueeze(1)
        # scores -- .shape: (batch_size, n_samples)
        scores = torch.sum(pred * sample_embed, dim=-1) + bias
        return scores


class RotatE(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.rel_fc = nn.Linear(configs.embed_dim, configs.embed_dim // 2)
        self.bias = nn.Parameter(torch.zeros(self.configs.n_ent))
        self.bias.requires_grad = False

    def forward(self, head_emb, rel_emb):
        rel_emb = self.rel_fc(rel_emb)
        head_emb_re, head_emb_im = torch.chunk(head_emb, 2, dim=1)
        rel_emb_re, rel_emb_im = torch.cos(rel_emb), torch.sin(rel_emb)
        pred_emb_re, pred_emb_im = self.hadamard_complex(
            head_emb_re, head_emb_im, rel_emb_re, rel_emb_im
        )
        return pred_emb_re, pred_emb_im

    def get_logits(self, pred, ent_embed):
        ent_emb_re, ent_emb_im = torch.chunk(ent_embed, 2, dim=1)
        diff_re, diff_im = self.pairwise_diff_complex(
            pred[0], pred[1], ent_emb_re, ent_emb_im
        )
        diff_abs = self.abs_complex(diff_re, diff_im)
        logits = self.configs.loss_gamma - self.norm_nonnegative(diff_abs, dim=2, p=1.)
        return logits

    # pred[0], pred[1] -- .shape: (batch_size, embed_dim // 2)
    # sample_embed .shape: (batch_size, n_samples, embed_dim)
    # sample_ids .shape: (batch_size, n_samples)
    def score_fn(self, pred, sample_embed, bias):
        # sample_emb_re, sample_emb_im -- .shape: (batch_szie, n_samples, embed_dim // 2)
        sample_emb_re, sample_emb_im = torch.chunk(sample_embed, 2, dim=2)
        # diff_re, diff_im -- .shape: (batch_size, n_samples, embed_dim // 2)
        diff_re, diff_im = self.pairwise_diff_complex(
            pred[0], pred[1], sample_emb_re, sample_emb_im
        )
        # diff_abs -- .shape: (batch_size, n_samples, embed_dim)
        diff_abs = self.abs_complex(diff_re, diff_im)
        logits = self.configs.loss_gamma - self.norm_nonnegative(diff_abs, dim=2, p=1.)
        return logits


    @staticmethod
    def hadamard_complex(x_re, x_im, y_re, y_im):
        "Hadamard product for complex vectors"
        result_re = x_re * y_re - x_im * y_im
        result_im = x_re * y_im + x_im * y_re
        return result_re, result_im


    def pairwise_diff_complex(self, x_re, x_im, y_re, y_im):
        "Pairwise difference of complex vectors"
        return self.pairwise_diff(x_re, y_re), self.pairwise_diff(x_im, y_im)

    @staticmethod
    def pairwise_diff(X, Y):
        """Compute pairwise difference of rows of X and Y.

        Returns tensor of shape len(X) x len(Y) x dim."""
        return X.unsqueeze(1) - Y

    @staticmethod
    def abs_complex(x_re, x_im):
        "Compute magnitude of given complex numbers"
        x_re_im = torch.stack((x_re, x_im), dim=0)  # dim0: real, imaginary
        return torch.norm(x_re_im, dim=0)  # sqrt(real^2+imaginary^2)

    @staticmethod
    def norm_nonnegative(x, dim: int, p: float):
        "Computes lp-norm along dim assuming that all inputs are non-negative."
        if p == 1.0:
            # speed up things for this common case. We known that the inputs are
            # non-negative here.
            return torch.sum(x, dim=dim)
        else:
            return torch.norm(x, dim=dim, p=p)
