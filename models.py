import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self, emb_size, num_lit_size, txt_lit_size):
        super(Gate, self).__init__()

        self.gate_ent = nn.Linear(emb_size, emb_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(emb_size))

        if num_lit_size == 0 and txt_lit_size == 0:
            raise ValueError("At least one of the literal sizes must be non-zero.")
        elif txt_lit_size == 0:
            # Only numerical literals
            self.gate_num_lit = nn.Linear(num_lit_size, emb_size, bias=False)
            self.g = nn.Linear(emb_size + num_lit_size, emb_size)
            self.mode = "num"
            print("Gate mode: num")
        elif num_lit_size == 0:
            # Only text literals
            self.gate_txt_lit = nn.Linear(txt_lit_size, emb_size, bias=False)
            self.g = nn.Linear(emb_size + txt_lit_size, emb_size)
            self.mode = "txt"
            print("Gate mode: txt")
        else:
            # Both
            self.gate_num_lit = nn.Linear(num_lit_size, emb_size, bias=False)
            self.gate_txt_lit = nn.Linear(txt_lit_size, emb_size, bias=False)
            self.g = nn.Linear(emb_size + num_lit_size + txt_lit_size, emb_size)
            self.mode = "all"
            print("Gate mode: all")

    def forward(self, x_ent, x_lit_num, x_lit_txt):
        if self.mode == "num":
            z = torch.sigmoid(self.gate_ent(x_ent) + self.gate_num_lit(x_lit_num) + self.gate_bias)
            h = torch.tanh(self.g(torch.cat([x_ent, x_lit_num], 1)))
        elif self.mode == "txt":
            z = torch.sigmoid(self.gate_ent(x_ent) + self.gate_txt_lit(x_lit_txt) + self.gate_bias)
            h = torch.tanh(self.g(torch.cat([x_ent, x_lit_txt], 1)))
        else:
            z = torch.sigmoid(
                self.gate_ent(x_ent) + self.gate_num_lit(x_lit_num) + self.gate_txt_lit(x_lit_txt) + self.gate_bias)
            h = torch.tanh(self.g(torch.cat([x_ent, x_lit_num, x_lit_txt], 1)))

        output = (1 - z) * x_ent + z * h

        return output


class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, lit_mode="none", numerical_literals=None,
                 text_literals=None, dropout=0.2, batch_norm=False, reg_weight=0.0):
        super(DistMult, self).__init__()

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.lit_mode = lit_mode
        if self.lit_mode == "all":
            print("Literal mode: all")
            self.numerical_literals = numerical_literals
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], text_literals.shape[1])
        elif self.lit_mode == "num":
            print("Literal mode: num")
            self.numerical_literals = numerical_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], 0)
        elif self.lit_mode == "txt":
            print("Literal mode: txt")
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, 0, text_literals.shape[1])

        self.dp = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_entity = nn.BatchNorm1d(embedding_dim)
            self.bn_relation = nn.BatchNorm1d(embedding_dim)
        self.reg_weight = reg_weight

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

        if self.lit_mode == "all":
            e1_num_lit = self.numerical_literals[e1_idx]
            e1_text_lit = self.text_literals[e1_idx]
            e1_emb = self.literal_embeddings(e1_emb, e1_num_lit, e1_text_lit)
            e2_num_lit = self.numerical_literals[e2_idx]
            e2_text_lit = self.text_literals[e2_idx]
            e2_emb = self.literal_embeddings(e2_emb, e2_num_lit, e2_text_lit)
        elif self.lit_mode == "num":
            e1_num_lit = self.numerical_literals[e1_idx]
            e1_emb = self.literal_embeddings(e1_emb, e1_num_lit, None)
            e2_num_lit = self.numerical_literals[e2_idx]
            e2_emb = self.literal_embeddings(e2_emb, e2_num_lit, None)
        elif self.lit_mode == "txt":
            e1_text_lit = self.text_literals[e1_idx]
            e1_emb = self.literal_embeddings(e1_emb, None, e1_text_lit)
            e2_text_lit = self.text_literals[e2_idx]
            e2_emb = self.literal_embeddings(e2_emb, None, e2_text_lit)

        # Apply dropout
        e1_emb = self.dp(e1_emb)
        r_emb = self.dp(r_emb)
        e2_emb = self.dp(e2_emb)

        # Calculate DistMult score
        out = torch.sigmoid(torch.sum(e1_emb * r_emb * e2_emb, dim=1))
        out = torch.flatten(out)

        # Regularization
        reg = 0
        if self.reg_weight != 0:
            reg = (torch.mean(e1_emb ** 2) + torch.mean(r_emb ** 2) + torch.mean(e2_emb ** 2)) / 3

        return out, reg


class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, lit_mode="none", numerical_literals=None,
                 text_literals=None, dropout=0.2, batch_norm=False, reg_weight=0.0):
        super(ComplEx, self).__init__()
        self.entity_embedding_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embedding_img = nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding_real = nn.Embedding(num_relations, embedding_dim)
        self.rel_embedding_img = nn.Embedding(num_relations, embedding_dim)

        self.lit_mode = lit_mode
        if self.lit_mode == "all":
            print("Literal mode: all")
            self.numerical_literals = numerical_literals
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], text_literals.shape[1])
        elif self.lit_mode == "num":
            print("Literal mode: num")
            self.numerical_literals = numerical_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], 0)
        elif self.lit_mode == "txt":
            print("Literal mode: txt")
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, 0, text_literals.shape[1])

        self.dp = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_entity = nn.BatchNorm1d(embedding_dim)
            self.bn_relation = nn.BatchNorm1d(embedding_dim)
        self.reg_weight = reg_weight

    def init(self):
        nn.init.xavier_normal_(self.emb_e_real.weight.data)
        nn.init.xavier_normal_(self.emb_e_img.weight.data)
        nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        nn.init.xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1_idx, rel_idx, e2_idx):
        # Embeddings
        e1_emb_real = self.entity_embedding_real(e1_idx)
        e1_emb_img = self.entity_embedding_img(e1_idx)
        r_emb_real = self.rel_embedding_real(rel_idx)
        r_emb_img = self.rel_embedding_img(rel_idx)
        e2_emb_real = self.entity_embedding_real(e2_idx)
        e2_emb_img = self.entity_embedding_img(e2_idx)

        # Batch normalization # TODO: Before or after literal embeddings?
        if self.batch_norm:
            e1_emb_real = self.bn_entity(e1_emb_real)
            e1_emb_img = self.bn_entity(e1_emb_img)
            r_emb_real = self.bn_relation(r_emb_real)
            r_emb_img = self.bn_relation(r_emb_img)
            e2_emb_real = self.bn_entity(e2_emb_real)
            e2_emb_img = self.bn_entity(e2_emb_img)

        if self.lit_mode == "all":
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
        elif self.lit_mode == "num":
            # Get embeddings for numerical literals
            # For e1
            e1_num_lit = self.numerical_literals[e1_idx]
            e1_emb_real = self.literal_embeddings(e1_emb_real, e1_num_lit, None)
            e1_emb_img = self.literal_embeddings(e1_emb_img, e1_num_lit, None)
            # For e2
            e2_num_lit = self.numerical_literals[e2_idx]
            e2_emb_real = self.literal_embeddings(e2_emb_real, e2_num_lit, None)
            e2_emb_img = self.literal_embeddings(e2_emb_img, e2_num_lit, None)
        elif self.lit_mode == "txt":
            # Get embeddings for textual literals
            # For e1
            e1_text_lit = self.text_literals[e1_idx]
            e1_emb_real = self.literal_embeddings(e1_emb_real, None, e1_text_lit)
            e1_emb_img = self.literal_embeddings(e1_emb_img, None, e1_text_lit)
            # For e2
            e2_text_lit = self.text_literals[e2_idx]
            e2_emb_real = self.literal_embeddings(e2_emb_real, None, e2_text_lit)
            e2_emb_img = self.literal_embeddings(e2_emb_img, None, e2_text_lit)


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
        reg = 0.0
        if self.reg_weight != 0:
            reg = (torch.mean(e1_emb_real ** 2) + torch.mean(e1_emb_img ** 2) + torch.mean(r_emb_real ** 2) +
                   torch.mean(r_emb_img ** 2) + torch.mean(e2_emb_real ** 2) + torch.mean(e2_emb_img ** 2)) / 6

        return out, reg


class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        x = x.view(n, -1)
        return x


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, lit_mode="none", numerical_literals=None,
                 text_literals=None, dropout=0.2, reg_weight=0.0, bias=False):
        super(ConvE, self).__init__()
        self.emb_w = 10
        self.emb_h = 20
        self.kernel_size = 3
        self.conv_channels = 32

        embedding_dim = self.emb_w * self.emb_h
        flattened_size = (self.emb_w * 2 - self.kernel_size + 1) * \
                         (self.emb_h - self.kernel_size + 1) * self.conv_channels # = 10368

        self.entity_embedding = torch.nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding = torch.nn.Embedding(num_relations, embedding_dim)

        self.conv_e = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=1, out_channels=self.conv_channels, kernel_size=self.kernel_size, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_channels),
            nn.Dropout2d(dropout),

            Flatten(),

            nn.Linear(in_features=flattened_size, out_features=embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
        )

        self.lit_mode = lit_mode
        if self.lit_mode == "all":
            print("Literal mode: all")
            self.numerical_literals = numerical_literals
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], text_literals.shape[1])
        elif self.lit_mode == "num":
            print("Literal mode: num")
            self.numerical_literals = numerical_literals
            self.literal_embeddings = Gate(embedding_dim, numerical_literals.shape[1], 0)
        elif self.lit_mode == "txt":
            print("Literal mode: txt")
            self.text_literals = text_literals
            self.literal_embeddings = Gate(embedding_dim, 0, text_literals.shape[1])

        self.reg_weight = reg_weight

    def init(self):
        nn.init.xavier_normal_(self.entity_embedding.weight.data)
        nn.init.xavier_normal_(self.rel_embedding.weight.data)

    def forward(self, e1_idx, rel_idx, e2_idx):
        e1_emb = self.entity_embedding(e1_idx)
        rel_emb = self.rel_embedding(rel_idx)
        e2_emb = self.entity_embedding(e2_idx)

        if self.lit_mode == "all":
            e1_num_lit = self.numerical_literals[e1_idx]
            e1_text_lit = self.text_literals[e1_idx]
            e1_emb = self.literal_embeddings(e1_emb, e1_num_lit, e1_text_lit)
            e2_num_lit = self.numerical_literals[e2_idx]
            e2_text_lit = self.text_literals[e2_idx]
            e2_emb = self.literal_embeddings(e2_emb, e2_num_lit, e2_text_lit)
        elif self.lit_mode == "num":
            e1_num_lit = self.numerical_literals[e1_idx]
            e1_emb = self.literal_embeddings(e1_emb, e1_num_lit, None)
            e2_num_lit = self.numerical_literals[e2_idx]
            e2_emb = self.literal_embeddings(e2_emb, e2_num_lit, None)
        elif self.lit_mode == "txt":
            e1_text_lit = self.text_literals[e1_idx]
            e1_emb = self.literal_embeddings(e1_emb, None, e1_text_lit)
            e2_text_lit = self.text_literals[e2_idx]
            e2_emb = self.literal_embeddings(e2_emb, None, e2_text_lit)

        # Reshape embeddings
        e1_emb = e1_emb.view(-1, self.emb_w, self.emb_h)
        rel_emb = rel_emb.view(-1, self.emb_w, self.emb_h)

        stacked_inputs = torch.cat([e1_emb, rel_emb], dim=1).unsqueeze(1)

        x = self.conv_e(stacked_inputs)
        x = torch.sum(x * e2_emb, dim=1) # TODO: Is this correct? Original code: torch.mm(x, e2_emb.t()) results in 256x256 matrix. Like this it is 256x1
        out = torch.sigmoid(x)

        # Regularization
        reg = 0.0
        if self.reg_weight != 0:
            reg = (torch.mean(e1_emb ** 2) + torch.mean(rel_emb ** 2) + torch.mean(e2_emb ** 2)) / 3

        return out, reg
