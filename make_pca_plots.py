import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset import LiteralLinkPredDataset
import os.path as osp
import matplotlib.pyplot as plt
import models

COLORS = {
    'human': '#a6cee3',
    'film': '#1f78b4',
    'city': '#b2df8a',
    'sports team': '#33a02c',
    'university': '#fb9a99',
    'music genre': '#e31a1c',
    'class of award': '#fdbf6f',
    'business': '#ff7f00',
    'country': '#cab2d6',
    'musical group': '#6a3d9a',
    'profession': '#ffff99',
    'television series': '#b15928'
}

ENTITY_LABELS_FILE_NAME = "final_mapping.csv"
DATASET_NAME = 'FB15k-237'

DISTMULT_MODEL_FILE_NAME = "08-13_07-08-58_DistMult_fb15k-237_model.pt"
DISTMULT_LIT_NUM_MODEL_FILE_NAME = "08-05_11-57-15_DistMultLit_fb15k-237_model.pt"
DISTMULT_LIT_TXT_MODEL_FILE_NAME = "08-11_00-27-13_DistMultLit_fb15k-237_model.pt"
DISTMULT_LIT_ALL_MODEL_FILE_NAME = "08-11_00-27-14_DistMultLit_fb15k-237_model.pt"
DISTMULT_LIT_ATTR_MODEL_FILE_NAME = "08-05_12-05-02_DistMultLit_fb15k-237_model.pt"

MODEL = "DistMultLit"
MODE = "all"

MODEL_FILE_NAME = ""
if MODEL == "DistMultLit":
    if MODE == "num":
        MODEL_FILE_NAME = DISTMULT_LIT_NUM_MODEL_FILE_NAME
    elif MODE == "txt":
        MODEL_FILE_NAME = DISTMULT_LIT_TXT_MODEL_FILE_NAME
    elif MODE == "all":
        MODEL_FILE_NAME = DISTMULT_LIT_ALL_MODEL_FILE_NAME
    elif MODE == "attr":
        MODEL_FILE_NAME = DISTMULT_LIT_ATTR_MODEL_FILE_NAME
    else:
        raise ValueError("Mode must be 'num', 'attr', 'txt' or 'all'")
elif MODEL == "DistMult":
    MODEL_FILE_NAME = DISTMULT_MODEL_FILE_NAME
else:
    raise ValueError("Model must be 'DistMult' or 'DistMultLit'")



def plot_samples(samples):
    #
    # Do PCA with sample embeddings
    pca = TSNE(n_components=2)
    emb = pca.fit_transform(np.array([s[2] for s in samples]))

    # Plot PCA results with these colors for the classes
    fig, ax = plt.subplots()
    for i in range(len(samples)):
        # Transform embeddings to 2D
        emb2d = emb[i]
        ax.scatter(emb2d[0], emb2d[1], c=samples[i][3], label=samples[i][1])

    plt.show()


def get_literal_embedding(model, sample_entities, mode="num"):
    entity_embeddings = model["entity_embeddings.weight"].detach()

    if mode == "num" or mode == "attr":
        state_dict = {
            "gate_ent.weight": model["literal_embeddings.gate_ent.weight"].detach(),
            "gate_num_lit.weight": model["literal_embeddings.gate_num_lit.weight"].detach(),
            "gate_bias": model["literal_embeddings.gate_bias"].detach(),
            "g.weight": model["literal_embeddings.g.weight"].detach(),
            "g.bias": model["literal_embeddings.g.bias"].detach()
        }

        lit_model = models.Gate(emb_size=entity_embeddings.shape[1], num_lit_size=dataset.literals_num.shape[1],
                                txt_lit_size=0)
    elif mode == "txt":
        state_dict = {
            "gate_ent.weight": model["literal_embeddings.gate_ent.weight"].detach(),
            "gate_txt_lit.weight": model["literal_embeddings.gate_txt_lit.weight"].detach(),
            "gate_bias": model["literal_embeddings.gate_bias"].detach(),
            "g.weight": model["literal_embeddings.g.weight"].detach(),
            "g.bias": model["literal_embeddings.g.bias"].detach()
        }

        lit_model = models.Gate(emb_size=entity_embeddings.shape[1], num_lit_size=0,
                                txt_lit_size=dataset.literals_txt.shape[1])
    elif mode == "all":
        state_dict = {
            "gate_ent.weight": model["literal_embeddings.gate_ent.weight"].detach(),
            "gate_num_lit.weight": model["literal_embeddings.gate_num_lit.weight"].detach(),
            "gate_txt_lit.weight": model["literal_embeddings.gate_txt_lit.weight"].detach(),
            "gate_bias": model["literal_embeddings.gate_bias"].detach(),
            "g.weight": model["literal_embeddings.g.weight"].detach(),
            "g.bias": model["literal_embeddings.g.bias"].detach()
        }

        lit_model = models.Gate(emb_size=entity_embeddings.shape[1], num_lit_size=dataset.literals_num.shape[1],
                                txt_lit_size=dataset.literals_txt.shape[1])
    else:
        raise ValueError("Mode must be 'num', 'attr', 'txt' or 'all'")

    lit_model.load_state_dict(state_dict)
    lit_model.eval()

    samples = []
    # Get embeddings for all entities
    for row in sample_entities.itertuples():
        emb = entity_embeddings[row.entity_index]
        if mode == "num":
            num_lit = dataset.literals_num[row.entity_index]
            emb = lit_model(emb, num_lit, None)
        elif mode == "attr":
            attr_lit = dataset.attr_relations_num[row.entity_index]
            emb = lit_model(emb, attr_lit, None)
        elif mode == "txt":
            txt_lit = dataset.literals_txt[row.entity_index]
            emb = lit_model(emb, None, txt_lit)
        elif mode == "all":
            num_lit = dataset.literals_num[row.entity_index]
            txt_lit = dataset.literals_txt[row.entity_index]
            emb = lit_model(emb, num_lit, txt_lit)
        color = COLORS[row.class_label]
        samples.append((row.entity_index, row.class_label, emb.detach().numpy(), color))

    return samples


if __name__ == "__main__":

    # Load entity labels (.csv file)
    entity_labels = pd.read_csv(f"data/FB15k-237/{ENTITY_LABELS_FILE_NAME}", sep=";")

    # Drop entities with class_label "other"
    entity_labels = entity_labels[entity_labels['class_label'] != 'other']

    # Load dataset
    if not osp.isfile(f'data/{DATASET_NAME}/processed.pt'):
        print('Process dataset...')
        dataset = LiteralLinkPredDataset(f'data/{DATASET_NAME}')
        torch.save(dataset, f'data/{DATASET_NAME}/processed.pt')

    dataset = torch.load(f'data/{DATASET_NAME}/processed.pt')

    # Add column to dataframe with entity index from dataset
    entity_labels['entity_index'] = entity_labels['dataset_entity'].apply(lambda x: dataset.entity2id[x])

    print(entity_labels.head())

    # Sample entities from each label
    sample_size = 50
    sample_entities = entity_labels.groupby('class_label').apply(lambda x: x.sample(sample_size)).reset_index(drop=True)
    sample_entities = sample_entities.sort_values(by='class_label')

    # Load model
    model = torch.load(f"results/{MODEL_FILE_NAME}", map_location=torch.device('cpu'))
    print(model.keys())

    # Get embeddings for all entities
    samples = []
    if MODEL == "DistMult":
        print("Getting embeddings for DistMult model...")
        entity_embeddings = model["entity_embeddings.weight"].detach().numpy()
        print(entity_embeddings.shape)
        for row in sample_entities.itertuples():
            emb = entity_embeddings[row.entity_index]
            color = COLORS[row.class_label]
            samples.append((row.entity_index, row.class_label, emb, color))

    elif MODEL == "DistMultLit":
        print("Getting embeddings for DistMultLit model...")
        samples = get_literal_embedding(model, sample_entities, mode=MODE)

    else:
        raise ValueError("Model not recognized.")

    plot_samples(samples)
