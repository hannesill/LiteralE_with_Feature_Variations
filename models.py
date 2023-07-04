import torch
from torch import nn


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
class DistMult(nn.Module):

    def __init__(self, num_entities, num_relations, embedding_dim, lit=False, numerical_literals=None,
                 text_literals=None, dropout=0.2, batch_norm=False, reg=False):
        super(DistMult, self).__init__()

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.lit = lit
        if self.lit:
            self.numerical_literals = numerical_literals
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], text_literals.shape[1])

        self.dp = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_entity = nn.BatchNorm1d(embedding_dim)
            self.bn_relation = nn.BatchNorm1d(embedding_dim)
        self.reg = reg

    def init_weights(self):
        # TODO: Why never called? Would calling it make sense?
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        nn.init.xavier_normal_(self.relation_embeddings.weight.data)

    def forward(self, e1_idx, r_idx, e2_idx):
        # Get embeddings with dropout
        e1_emb = self.entity_embeddings(e1_idx)
        r_emb = self.relation_embeddings(r_idx)
        e2_emb = self.entity_embeddings(e2_idx)

        # Batch normalization # TODO: Before or after literal embeddings?
        if self.batch_norm:
            e1_emb = self.bn_entity(e1_emb)
            r_emb = self.bn_relation(r_emb)
            e2_emb = self.bn_entity(e2_emb)

        if self.lit:
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

        # Calculate DistMult score
        out = torch.sigmoid(torch.sum(e1_emb * r_emb * e2_emb, dim=1))
        out = torch.flatten(out)

        # Regularization
        if self.reg:
            reg = (torch.mean(e1_emb ** 2) + torch.mean(r_emb ** 2) + torch.mean(e2_emb ** 2)) / 3
        else:
            reg = 0.0

        return out, reg


class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, lit=False, numerical_literals=None,
                 text_literals=None, dropout=0.2, batch_norm=False, reg=False):
        super(ComplEx, self).__init__()
        self.entity_embedding_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embedding_img = nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding_real = nn.Embedding(num_relations, embedding_dim)
        self.rel_embedding_img = nn.Embedding(num_relations, embedding_dim)

        self.lit = lit
        if self.lit:
            self.numerical_literals = numerical_literals
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], text_literals.shape[1])

        self.dp = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_entity = nn.BatchNorm1d(embedding_dim)
            self.bn_relation = nn.BatchNorm1d(embedding_dim)
        self.reg = reg

    def init(self):
        nn.init.xavier_normal_(self.emb_e_real.weight.data)
        nn.init.xavier_normal_(self.emb_e_img.weight.data)
        nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        nn.init.xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1_idx, rel_idx, e2_idx):
        # Embeddings
        e1_emb_real = self.entity_embedding(e1_idx)
        e1_emb_img = self.entity_embedding(e1_idx)
        r_emb_real = self.rel_embedding(rel_idx)
        r_emb_img = self.rel_embedding(rel_idx)
        e2_emb_real = self.entity_embedding(e2_idx)
        e2_emb_img = self.entity_embedding(e2_idx)

        # Batch normalization # TODO: Before or after literal embeddings?
        if self.batch_norm:
            e1_emb_real = self.bn_entity(e1_emb_real)
            e1_emb_img = self.bn_entity(e1_emb_img)
            r_emb_real = self.bn_relation(r_emb_real)
            r_emb_img = self.bn_relation(r_emb_img)
            e2_emb_real = self.bn_entity(e2_emb_real)
            e2_emb_img = self.bn_entity(e2_emb_img)

        if self.lit:
            # Get embeddings for numerical and text literals
            # For e1
            e1_num_lit = self.numerical_literals[e1_idx]
            e1_text_lit = self.text_literals[e1_idx]
            e1_emb_real = self.literal_embeddings(e1_emb_real, e1_num_lit, e1_text_lit)
            e1_emb_img = self.literal_embeddings(e1_emb_img, e1_num_lit, e1_text_lit)
            # For e2
            e2_num_lit = self.numerical_literals[e2_idx]
            e2_text_lit = self.text_literals[e2_idx]
            e2_emb_real = self.literal_embeddings(e2_emb_real, e2_num_lit, e2_text_lit)
            e2_emb_img = self.literal_embeddings(e2_emb_img, e2_num_lit, e2_text_lit)

        # Apply dropout
        e1_emb_real = self.dp(e1_emb_real)
        e1_emb_img = self.dp(e1_emb_img)
        r_emb_real = self.dp(r_emb_real)
        r_emb_img = self.dp(r_emb_img)
        e2_emb_real = self.dp(e2_emb_real)
        e2_emb_img = self.dp(e2_emb_img)

        # Calculate ComplEx score
        real_real_real = torch.sum(e1_emb_real * r_emb_real * e2_emb_real, dim=1)
        real_img_img = torch.sum(e1_emb_real * r_emb_img * e2_emb_img, dim=1)
        img_real_img = torch.sum(e1_emb_img * r_emb_real * e2_emb_img, dim=1)
        img_img_real = torch.sum(e1_emb_img * r_emb_img * e2_emb_real, dim=1)

        out = torch.sigmoid(real_real_real + real_img_img + img_real_img - img_img_real)
        out = torch.flatten(out)

        # Regularization
        if self.reg:
            reg = (torch.mean(e1_emb_real ** 2) + torch.mean(e1_emb_img ** 2) + torch.mean(r_emb_real ** 2) +
                   torch.mean(r_emb_img ** 2) + torch.mean(e2_emb_real ** 2) + torch.mean(e2_emb_img ** 2)) / 6
        else:
            reg = 0.0

        return out, reg


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, lit=False, numerical_literals=None,
                 text_literals=None, dropout=0.2, reg=False, bias=True):
        super(ConvE, self).__init__()
        self.entity_embedding = torch.nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding = torch.nn.Embedding(num_relations, embedding_dim)

        self.lit = lit
        if self.lit:
            self.numerical_literals = numerical_literals
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], text_literals.shape[1])

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=bias)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368, embedding_dim)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)

        self.dp = torch.nn.Dropout(dropout)
        self.reg = reg

    def init(self):
        nn.init.xavier_normal_(self.entity_embedding.weight.data)
        nn.init.xavier_normal_(self.rel_embedding.weight.data)

    def forward(self, e1_idx, rel_idx, e2_idx):
        # TODO: Woher hat LiteralE code die 10 und 20? In ConvE Implementierung stehen da emb_dim_1 und emb_dim_2.
        #  Da wäre aber die Frage was sind das für Dimensionen?
        e1_emb = self.entity_embedding(e1_idx).view(-1, 1, 10, 20)
        rel_emb = self.rel_embedding(rel_idx).view(-1, 1, 10, 20)
        e2_emb = self.entity_embedding(e2_idx).view(-1, 1, 10, 20)

        if self.lit:
            # Get embeddings for numerical and text literals
            # For e1
            e1_num_lit = self.numerical_literals[e1_idx]
            e1_text_lit = self.text_literals[e1_idx]
            e1_emb = self.literal_embeddings(e1_emb, e1_num_lit, e1_text_lit)
            # For e2
            e2_num_lit = self.numerical_literals[e2_idx]
            e2_text_lit = self.text_literals[e2_idx]
            e2_emb = self.literal_embeddings(e2_emb, e2_num_lit, e2_text_lit)

        stacked_inputs = torch.cat([e1_emb, rel_emb], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.sum(x * e2_emb, dim=1)  # TODO: Passt das anstelle von: torch.mm(x, self.entity_embedding.weight.transpose(1, 0)) ?
        x += self.b.expand_as(x)
        out = torch.sigmoid(x)

        # Regularization
        if self.reg:
            reg = (torch.mean(e1_emb ** 2) + torch.mean(rel_emb ** 2) + torch.mean(e2_emb ** 2)) / 3
        else:
            reg = 0.0

        return out, reg
