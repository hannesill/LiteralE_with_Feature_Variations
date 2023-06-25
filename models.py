import torch
from torch import nn


class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, dropout=0.2, batch_norm=False, reg=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding = nn.Embedding(num_relations, embedding_dim)

        self.dp_ent = nn.Dropout(dropout)
        self.dp_rel = nn.Dropout(dropout)
        self.reg = reg

        self.bn_head = nn.BatchNorm1d(embedding_dim)
        self.bn_rel = nn.BatchNorm1d(embedding_dim)
        self.bn_tail = nn.BatchNorm1d(embedding_dim)

    def init(self):
        nn.init.xavier_normal_(self.entity_embedding.weight.data)
        nn.init.xavier_normal_(self.rel_embedding.weight.data)

    def forward(self, e1_idx, rel_idx, e2_idx):
        e1_emb = self.dp_ent(self.entity_embedding(e1_idx))
        r_emb = self.dp_rel(self.rel_embedding(rel_idx))
        e2_emb = self.dp_ent(self.entity_embedding(e2_idx))

        if self.batch_norm:
            e1_emb = self.bn_head(e1_emb)
            r_emb = self.bn_rel(r_emb)
            e2_emb = self.bn_tail(e2_emb)

        out = torch.sigmoid(torch.sum(e1_emb * r_emb * e2_emb, dim=1))
        out = torch.flatten(out)

        if self.reg:
            # Herre we don't use reg, but we could
            reg = 0.0
        else:
            reg = 0.0

        return out, reg


##############################################################################################################
# DistMultLit
##############################################################################################################
class Gate(nn.Module):
    def __init__(self, emb_size, num_lit_size, txt_lit_size):
        super(Gate, self).__init__()

        self.gate_ent = nn.Linear(emb_size, emb_size, bias=False)
        self.gate_num_lit = nn.Linear(num_lit_size, emb_size, bias=False)
        self.gate_txt_lit = nn.Linear(txt_lit_size, emb_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(emb_size))
        self.g = nn.Linear(emb_size + num_lit_size + txt_lit_size, emb_size)

    def forward(self, x_ent, x_lit_num, x_lit_txt):
        z = torch.sigmoid(
            self.gate_ent(x_ent) + self.gate_num_lit(x_lit_num) + self.gate_txt_lit(x_lit_txt) + self.gate_bias)
        h = torch.tanh(self.g(torch.cat([x_ent, x_lit_num, x_lit_txt], 1)))

        output = (1 - z) * x_ent + z * h

        return output


# TODO (optional): early stopping
class DistMultLit(nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, text_literals, embedding_dim,
                 dropout=0.2, batch_norm=False, reg=False):
        super(DistMultLit, self).__init__()
        # Initialize parameters
        self.batch_norm = batch_norm
        self.reg = reg
        self.numerical_literals = numerical_literals
        self.text_literals = text_literals

        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], text_literals.shape[1])

        # Initialize loss function and weights
        self.loss = nn.BCELoss()
        # self.init_weights()
        # Difference: Paper model does not call init_weights. TODO: Why?

        # Initialize dropout and batch normalization
        # TODO: Dropout auch für Gate?
        self.dp = nn.Dropout(dropout)

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(embedding_dim)

    def init_weights(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        nn.init.xavier_normal_(self.relation_embeddings.weight.data)

    def forward(self, e1_idx, r_idx, e2_idx):
        # Get embeddings for entities and relations
        e1_emb = self.entity_embeddings(e1_idx)
        r_emb = self.relation_embeddings(r_idx)
        e2_emb = self.entity_embeddings(e2_idx)

        # Batch normalization
        # TODO: Batch normalization for both e1 and e2? Also for literal embeddings?
        if self.batch_norm:
            e1_emb = self.bn(e1_emb)
            r_emb = self.bn(r_emb)
            e2_emb = self.bn(e2_emb)

        # Get embeddings for numerical and text literals
        # For e1
        e1_num_lit = self.numerical_literals[e1_idx]
        e1_text_lit = self.text_literals[e1_idx]
        e1_emb = self.literal_embeddings(e1_emb, e1_num_lit, e1_text_lit)
        # For e2
        e2_num_lit = self.numerical_literals[e2_idx]
        e2_text_lit = self.text_literals[e2_idx]
        e2_emb = self.literal_embeddings(e2_emb, e2_num_lit, e2_text_lit)

        # Apply dropout
        e1_emb = self.dp(e1_emb)
        r_emb = self.dp(r_emb)
        e2_emb = self.dp(e2_emb)

        out = torch.sigmoid(torch.sum(e1_emb * r_emb * e2_emb, dim=1))
        out = torch.flatten(out)

        if self.reg:
            reg = (torch.mean(e1_emb ** 2) + torch.mean(r_emb ** 2) + torch.mean(e2_emb ** 2)) / 3
        else:
            reg = 0.0

        return out, reg
