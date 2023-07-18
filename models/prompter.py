import torch.nn as nn


class Prompter(nn.Module):
    def __init__(self, plm_config, embed_dim, prompt_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = plm_config.hidden_size
        self.num_heads = plm_config.num_attention_heads
        self.num_layers = plm_config.num_hidden_layers
        self.prompt_length = prompt_length
        self.prompt_hidden_dim = plm_config.prompt_hidden_dim
        self.dropout = nn.Dropout(plm_config.hidden_dropout_prob)
        self.seq = nn.Sequential(
            nn.Linear(embed_dim, self.prompt_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.prompt_hidden_dim, self.model_dim * self.num_layers * prompt_length, bias=False),
        )

    def forward(self, x):
        # out -- .shape: (batch_size, 2, model_dim * num_layers * prompt_length * 2)
        out = self.seq(x)
        out = out.view(x.size(0), self.num_layers, -1, self.model_dim)
        out = self.dropout(out)
        return out
