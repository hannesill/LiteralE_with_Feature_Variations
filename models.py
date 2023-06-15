import torch
from torch import nn
from torch.nn.init import xavier_normal_


class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, dropout=0.2, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.entity_embedding = torch.nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding = torch.nn.Embedding(num_relations, embedding_dim)

        self.dp_ent = torch.nn.Dropout(dropout)
        self.dp_rel = torch.nn.Dropout(dropout)

        self.bn_head = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_rel = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_tail = torch.nn.BatchNorm1d(embedding_dim)

    def init(self):
        xavier_normal_(self.entity_embedding.weight.data)
        xavier_normal_(self.rel_embedding.weight.data)

    def forward(self, head_idx, rel_idx, tail_idx):
        h_head = self.dp_ent(self.entity_embedding(head_idx))
        h_relation = self.dp_rel(self.rel_embedding(rel_idx))
        h_tail = self.dp_ent(self.entity_embedding(tail_idx))

        if self.batch_norm:
            h_head = self.bn_head(h_head)
            h_relation = self.bn_rel(h_relation)
            h_tail = self.bn_tail(h_tail)

        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))
        out = torch.flatten(out)

        return out


##############################################################################################################
# DistMultLit
##############################################################################################################
class Gate(nn.Module):

    def __init__(self, num_numerical_literals, num_text_literals, embedding_dim):
        super(Gate, self).__init__()
        self.gate_ent = nn.Linear(embedding_dim, embedding_dim)
        self.gate_num_lit = nn.Linear(num_numerical_literals, embedding_dim)
        self.gate_text_lit = nn.Linear(num_text_literals, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.W_h = nn.Linear(embedding_dim + num_numerical_literals + num_text_literals, embedding_dim)

    def forward(self, entity_emb, entity_num_lit, entity_text_lit):
        z_e = self.gate_ent(entity_emb)
        z_l_num = self.gate_num_lit(entity_num_lit)
        z_l_text = self.gate_text_lit(entity_text_lit)
        z = torch.sigmoid(z_e + z_l_num + z_l_text + self.bias)
        h = torch.tanh(self.W_h(torch.cat((entity_emb, entity_num_lit, entity_text_lit), dim=1)))

        return z * h + (1 - z) * entity_emb


# TODO (wieder einbauen): Regularization (nicht analysieren, bring nix)
# TODO (optional): early stopping
class DistMultLit(nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, text_literals, embedding_dim,
                 dropout=0.2, batch_norm=False):
        super(DistMultLit, self).__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.batch_norm = batch_norm
        self.numerical_literals = numerical_literals
        self.text_literals = text_literals

        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.literal_embeddings = Gate(numerical_literals.shape[1], text_literals.shape[1], embedding_dim)

        # Initialize loss function and weights
        self.loss = nn.BCELoss()
        self.init_weights()

        # Initialize dropout and batch normalization
        # TODO: Dropout auch für Gate?
        self.dp_e = torch.nn.Dropout(dropout)
        self.dp_r = torch.nn.Dropout(dropout)

        # TODO: Batch normalization for both e1 and e2? Also for literal embeddings?
        if self.batch_norm:
            self.bn_e1 = torch.nn.BatchNorm1d(embedding_dim)
            self.bn_r = torch.nn.BatchNorm1d(embedding_dim)
            self.bn_e2 = torch.nn.BatchNorm1d(embedding_dim)

    def init_weights(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        nn.init.xavier_normal_(self.relation_embeddings.weight.data)

    def forward(self, e1_idx, r_idx, e2_idx):
        # Get embeddings for entities and relations with dropout
        e1_emb = self.dp_e(self.entity_embeddings(e1_idx))
        r_emb = self.dp_r(self.relation_embeddings(r_idx))
        e2_emb = self.dp_e(self.entity_embeddings(e2_idx))

        # Batch normalization
        if self.batch_norm:
            e1_emb = self.bn_e1(e1_emb)
            r_emb = self.bn_r(r_emb)
            e2_emb = self.bn_e2(e2_emb)

        # Get embeddings for numerical and text literals
        # For e1
        e1_num_lit = self.numerical_literals[e1_idx]
        e1_text_lit = self.text_literals[e1_idx]
        e1_emb = self.literal_embeddings(e1_emb, e1_num_lit, e1_text_lit)
        # For e2
        e2_num_lit = self.numerical_literals[e2_idx]
        e2_text_lit = self.text_literals[e2_idx]
        e2_emb = self.literal_embeddings(e2_emb, e2_num_lit, e2_text_lit)

        result = torch.sum(e1_emb * r_emb * e2_emb, dim=1)
        return torch.sigmoid(result)


##############################################################################################################
# Paper DistMultLit
##############################################################################################################

