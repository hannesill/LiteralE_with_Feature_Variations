from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import spacy


class LiteralLinkPredDataset(Dataset):

    def __getitem__(self, index):
        # placeholder
        return None

    def __init__(self, triple_file, embedding_dim=200, transform=None, target_transform=None):
        # Parameters
        self.triple_file = triple_file
        self.embedding_dim = embedding_dim
        self.transform = transform
        self.target_transform = target_transform

        # Data
        self.df_triples_train, self.df_triples_val, self.df_triples_test, self.df_literals_num, self.df_literals_txt \
            = self.load_dataframes()

        # Relational data
        self.entities, self.relations = self.load_relational_data()

        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

        self.entity2id = {self.entities[i]: i for i in range(self.num_entities)}
        self.relation2id = {self.relations[i]: i for i in range(self.num_relations)}

        self.edge_index_train, self.edge_index_val, self.edge_index_test = self.load_edge_indices()
        self.edge_type_train, self.edge_type_val, self.edge_type_test = self.load_edge_types()

        # Literal data
        self.literals_num = self.load_literals_num()
        self.literals_txt = self.load_literals_txt()

    def load_dataframes(self):
        print('start loading dataframes')
        df_triples_train = pd.read_csv(osp.join(self.triple_file, 'train.txt'), header=None, sep='\t')
        df_triples_val = pd.read_csv(osp.join(self.triple_file, 'valid.txt'), header=None, sep='\t')
        df_triples_test = pd.read_csv(osp.join(self.triple_file, 'test.txt'), header=None, sep='\t')
        df_literals_num = pd.read_csv(osp.join(self.triple_file, 'numerical_literals.txt'), header=None, sep='\t')
        df_literals_txt = pd.read_csv(osp.join(self.triple_file, 'text_literals.txt'), header=None, sep='\t')

        return df_triples_train, df_triples_val, df_triples_test, df_literals_num, df_literals_txt

    def load_relational_data(self):
        print('start loading relational data')
        self.entities = list(set(np.concatenate([self.df_triples_train[0].unique(),
                                                 self.df_triples_test[0].unique(),
                                                 self.df_triples_val[0].unique(),
                                                 self.df_triples_train[2].unique(),
                                                 self.df_triples_test[2].unique(),
                                                 self.df_triples_val[2].unique(),
                                                 self.df_literals_num[0].unique(),
                                                 self.df_literals_txt[0].unique()])))

        self.relations = list(set(np.concatenate([self.df_triples_train[1].unique(),
                                                  self.df_triples_test[1].unique(),
                                                  self.df_triples_val[1].unique()])))

        return self.entities, self.relations

    def load_edge_indices(self):
        edge_index_train = torch.stack([torch.tensor(self.df_triples_train[0].map(self.entity2id)),
                                        torch.tensor(self.df_triples_train[2].map(self.entity2id))])
        edge_index_val = torch.stack([torch.tensor(self.df_triples_val[0].map(self.entity2id)),
                                      torch.tensor(self.df_triples_val[2].map(self.entity2id))])
        edge_index_test = torch.stack([torch.tensor(self.df_triples_test[0].map(self.entity2id)),
                                       torch.tensor(self.df_triples_test[2].map(self.entity2id))])

        return edge_index_train, edge_index_val, edge_index_test

    def load_edge_types(self):
        edge_type_train = torch.tensor(self.df_triples_train[1].map(self.relation2id))
        edge_type_val = torch.tensor(self.df_triples_val[1].map(self.relation2id))
        edge_type_test = torch.tensor(self.df_triples_test[1].map(self.relation2id))

        return edge_type_train, edge_type_val, edge_type_test

    def load_literals_num(self):
        # with E = number of embeddings, R = number of attributive relations, V = feature dim
        print('start loading numerical literals: E x R')
        attr_relations_num_unique = list(self.df_literals_num[1].unique())
        attr_relation_2_id_num = {attr_relations_num_unique[i]: i for i in range(len(attr_relations_num_unique))}

        self.df_literals_num[0] = self.df_literals_num[0].map(self.entity2id).astype(int)
        self.df_literals_num[1] = self.df_literals_num[1].map(attr_relation_2_id_num).astype(int)
        self.df_literals_num[2] = self.df_literals_num[2].astype(float)

        num_attributive_relations_num = len(attr_relations_num_unique)

        # Extract numerical literal feature vectors for each entity for literal values AND attributive relations
        features_num = []
        features_num_mask = []
        features_num_attr = []
        features_num_attr_mask = []
        for i in tqdm(range(len(self.entities))):
            df_i = self.df_literals_num[self.df_literals_num[0] == i]

            feature_i = torch.zeros(num_attributive_relations_num)
            feature_i_mask = torch.zeros(num_attributive_relations_num, dtype=torch.bool)
            feature_i_attr = torch.zeros(num_attributive_relations_num)
            feature_i_attr_mask = torch.zeros(num_attributive_relations_num, dtype=torch.bool)
            for index, row in df_i.iterrows():
                feature_i[int(row[1])] = float(row[2])
                feature_i_mask[int(row[1])] = True

                # TODO: add to attributive relation feature

            features_num.append(feature_i)
            features_num_mask.append(feature_i_mask)
        features_num = torch.stack(features_num)
        features_num_mask = torch.stack(features_num_mask)

        max_lit, min_lit = torch.max(features_num, dim=0).values, torch.min(features_num, dim=0).values

        # Normalize numerical literals
        features_num = (features_num - min_lit) / (max_lit - min_lit + 1e-8)

        return features_num

    def load_literals_txt(self):
        print('start loading textual literals: E x R x V')
        attr_relations_txt = list(self.df_literals_txt[1].unique())
        attr_relation_2_id_txt = {attr_relations_txt[i]: i for i in range(len(attr_relations_txt))}

        self.df_literals_txt[0] = self.df_literals_txt[0].map(self.entity2id).astype(int)
        self.df_literals_txt[1] = self.df_literals_txt[1].map(attr_relation_2_id_txt).astype(int)
        self.df_literals_num[2] = self.df_literals_num[2].astype(str)
        num_attributive_relations_txt = len(attr_relations_txt)
        nlp = spacy.load('en_core_web_md')

        # TODO (optional): Take every literal feature and not just 1 -> take a look at LiteralE original impl
        # TODO implement to use the non negative filter features_num_mask -> Moritz
        features_txt = []
        for i in tqdm(range(len(self.entities))):
            df_i = self.df_literals_txt[self.df_literals_txt[0] == i]

            features_txt_i = torch.zeros(num_attributive_relations_txt, self.embedding_dim)
            for index, row in df_i.iterrows():
                # TODO (optional): BERT embeddings ausprobieren
                spacy_sentence = torch.tensor(nlp(row[2]).vector)
                features_txt_i[int(row[1])] = spacy_sentence

            features_txt.append(features_txt_i)

        features_txt = torch.stack(features_txt)

        return features_txt


if __name__ == '__main__':
    dataset = LiteralLinkPredDataset('./data/fb15k-237')
