import argparse
import json
from datetime import datetime
import torch
import os.path as osp
import numpy as np
import time
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataset import LiteralLinkPredDataset
from models import DistMult, DistMultLit


def negative_sampling(edge_index, num_nodes, eta=1):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(0) * eta) < 0.5
    mask_2 = ~mask_1

    mask_1 = mask_1.to(DEVICE)
    mask_2 = mask_2.to(DEVICE)

    neg_edge_index = edge_index.clone().repeat(eta, 1)
    neg_edge_index[mask_1, 0] = torch.randint(num_nodes, (1, mask_1.sum()), device=DEVICE)
    neg_edge_index[mask_2, 1] = torch.randint(num_nodes, (1, mask_2.sum()), device=DEVICE)

    return neg_edge_index


def train_standard_lp(config,
                      model_lp,
                      loss_function_model,
                      optimizer,
                      dataset):
    model_lp.train()
    start = time.time()

    train_edge_index_t = dataset.edge_index_train.t().to(DEVICE)
    train_edge_type = dataset.edge_type_train.to(DEVICE)

    edge_index_batches = torch.split(train_edge_index_t, config['batch_size'])
    edge_type_batches = torch.split(train_edge_type, config['batch_size'])

    batch_indices = np.arange(len(edge_index_batches))
    np.random.shuffle(batch_indices)

    loss_total = 0

    for batch_index in tqdm(batch_indices):
        edge_idxs, relation_idx = edge_index_batches[batch_index], edge_type_batches[batch_index]
        optimizer.zero_grad()

        edge_idxs_neg = negative_sampling(edge_idxs, dataset.num_entities, eta=config['eta'])

        out_pos = model_lp.forward(edge_idxs[:, 0], relation_idx, edge_idxs[:, 1])
        out_neg = model_lp.forward(edge_idxs_neg[:, 0], relation_idx.repeat(config['eta']), edge_idxs_neg[:, 1])

        out = torch.cat([out_pos, out_neg], dim=0)
        gt = torch.cat([torch.ones(len(relation_idx)), torch.zeros(len(relation_idx) * config['eta'])], dim=0).to(
            DEVICE)

        # print('size lp:', gt.size())

        loss = loss_function_model(out, gt)

        loss_total += loss.item()
        loss.backward()
        optimizer.step()

    end = time.time()
    print('elapsed time:', end - start)
    print('loss:', loss_total / len(edge_index_batches))


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    # TODO: pessimistic Teil raus nehmen oder zu pessimistic wie bei optimistic + 1 hinzuzufügen und testweise ausführen
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    # pessimistic = (ranks >= true).sum()
    # return (optimistic + pessimistic).float() * 0.5
    return optimistic.float() * 0.5


@torch.no_grad()
def compute_mrr_triple_scoring(model_lp, dataset, eval_edge_index, eval_edge_type,
                               fast=False):
    model_lp.eval()
    ranks = []
    num_samples = eval_edge_type.numel() if not fast else 5000
    for triple_index in tqdm(range(num_samples)):
        (src, dst), rel = eval_edge_index[:, triple_index], eval_edge_type[triple_index]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(dataset.num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (dataset.edge_index_train, dataset.edge_type_train),
            (dataset.edge_index_val, dataset.edge_type_val),
            (dataset.edge_index_test, dataset.edge_type_test),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(dataset.num_entities)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_typ_tensor = torch.full_like(tail, fill_value=rel).to(DEVICE)

        out = model_lp.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(dataset.num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (dataset.edge_index_train, dataset.edge_type_train),
            (dataset.edge_index_val, dataset.edge_type_val),
            (dataset.edge_index_test, dataset.edge_type_test),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(dataset.num_entities)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_typ_tensor = torch.full_like(head, fill_value=rel).to(DEVICE)

        out = model_lp.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        rank = compute_rank(out)
        ranks.append(rank)

    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)
    return (1. / ranks).mean().item(), \
        ranks.mean().item(), \
        (ranks[ranks <= 10].size(0) / num_ranks), \
        (ranks[ranks <= 5].size(0) / num_ranks), \
        (ranks[ranks <= 3].size(0) / num_ranks), \
        (ranks[ranks <= 1].size(0) / num_ranks)


def train_lp_objective(config, model_lp):
    dataset = config['dataset']

    loss_function_model = torch.nn.BCELoss(reduction='mean')
    if config['alpha'] > 0:
        optimizer = torch.optim.Adam(list(model_lp.parameters()), lr=config['lr'])
    else:
        optimizer = torch.optim.Adam(model_lp.parameters(), lr=config['lr'])

    start_epoch = 1
    model_lp.train()

    # History of validation metrics
    history = {"epoch": [],
               "val_mrr": [],
               "val_mr": [],
               "val_hits10": [],
               "val_hits5": [],
               "val_hits3": [],
               "val_hits1": []}

    for epoch in range(start_epoch, config['epochs'] + 1):
        print(f"--> Epoch {epoch}")
        train_standard_lp(config,
                          model_lp,
                          loss_function_model,
                          optimizer,
                          dataset)
        # Evaluating
        if epoch % config['val_every'] == 0:
            print("Evaluating model...")
            # TODO: Ggf nicht nur auf 5000 sondern auf allen Tripeln evaluieren
            mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model_lp,
                                                                              dataset,
                                                                              dataset.edge_index_val,
                                                                              dataset.edge_type_val,
                                                                              fast=True)
            print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

            # Save validation metrics
            history["epoch"].append(epoch)
            history["val_mrr"].append(mrr)
            history["val_mr"].append(mr)
            history["val_hits10"].append(hits10)
            history["val_hits5"].append(hits5)
            history["val_hits3"].append(hits3)
            history["val_hits1"].append(hits1)

            # Save history dictionary
            with open(f"results/{RUN_NAME}_history.json", "w+") as f:
                json.dump(history, f)

            # Save model
            torch.save(model_lp.state_dict(), f"results/{RUN_NAME}_model.pt")


if __name__ == '__main__':
    # Set random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        print('Using CUDA')
        DEVICE = torch.device('cuda')
    else:
        print('Using CPU')
        DEVICE = torch.device('cpu')

    dataset_name = 'fb15k-237'
    if not osp.isfile(f'data/{dataset_name}/processed.pt'):
        print('Process dataset...')
        dataset = LiteralLinkPredDataset(f'data/{dataset_name}')
        torch.save(dataset, f'data/{dataset_name}/processed.pt')
    print('Load processed dataset...')
    dataset = torch.load(f'data/{dataset_name}/processed.pt')

    parser = argparse.ArgumentParser()
    parser.add_argument("--lit", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=100)
    args = parser.parse_args()
    if args.lit:
        model_type = "DistMultLit"
    else:
        model_type = "DistMult"
    print(f"Model type: {model_type}")

    EPOCHS = args.epochs
    VAL_EVERY = args.val_every

    RUN_NAME = datetime.now().strftime("%m-%d_%H-%M") + "_" + model_type + "_" + dataset_name

    # default config (best for DistMult)
    config = {'dataset': dataset,
              'epochs': EPOCHS,
              'val_every': VAL_EVERY,
              'dim': 100,
              'lr': 0.00065,
              'batch_size': 256,
              'dropout': 0.2,
              'alpha': 0,
              'eta': 100,
              'reg': False,
              'batch_norm': False}

    # TODO: For DistMult+LiteralE config look at LiteralE paper

    # 14000, 1, 300 -> 14000, 300
    dataset.features_txt = dataset.features_txt.squeeze()

    # create model
    if model_type == "DistMultLit":
        model_lp = DistMultLit(dataset.num_entities,
                               dataset.num_relations,
                               dataset.features_num.to(DEVICE),
                               dataset.features_txt.to(DEVICE),
                               config['dim'])
    else:
        model_lp = DistMult(dataset.num_entities, dataset.num_relations, config['dim'])
    model_lp.to(DEVICE)

    print(dataset.num_entities, dataset.num_relations)
    print(dataset.features_num.shape, dataset.features_txt.shape)

    # train model
    print("Start training...")
    train_lp_objective(config, model_lp)

    # test model
    print("Start testing...")
    # TODO: Nicht nur auf 5000 sondern auf allen Tripeln evaluieren
    mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model_lp,
                                                                      dataset,
                                                                      dataset.edge_index_test,
                                                                      dataset.edge_type_test,
                                                                      fast=True)
    print('test mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

    # Save test results
    with open(f"results/{RUN_NAME}_test_results.txt", "w") as f:
        f.write(f"test mrr: {mrr}, mr: {mr}, hits@10: {hits10}, hits@5: {hits5}, hits@3: {hits3}, hits@1: {hits1}")

    # Load history dictionary
    with open(f"results/{RUN_NAME}_history.json", "r") as f:
        history = json.load(f)

    # Get validation metrics histories
    epochs = history["epoch"]
    mrr_history = history["val_mrr"]
    mr_history = history["val_mr"]
    hits10_history = history["val_hits10"]
    hits5_history = history["val_hits5"]
    hits3_history = history["val_hits3"]
    hits1_history = history["val_hits1"]

    print(history)

    # Plot all histories and save figures
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    for ax, history, title in zip(axs.flatten(),
                                  [mrr_history, mr_history, hits10_history, hits5_history, hits3_history, hits1_history],
                                  ["MRR", "MR", "Hits@10", "Hits@5", "Hits@3", "Hits@1"]):
        ax.plot(epochs, history)
        ax.set_title(f"{title} history")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
    plt.savefig(f"results/{RUN_NAME}_history.png")
