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


def negative_sampling(edge_idxs, num_nodes, eta=1):
    """
    Sample negative edges by corrupting either the subject or the object of each edge.

    :param edge_idxs: torch.tensor of shape (2, num_edges)
    :param num_nodes: int
    :param eta: int (default: 1) number of negative samples per positive sample
    :return: torch.tensor of shape (2 * eta, num_edges)
    """

    mask_1 = torch.rand(edge_idxs.size(0) * eta) < 0.5
    mask_2 = ~mask_1

    mask_1 = mask_1.to(DEVICE)
    mask_2 = mask_2.to(DEVICE)

    neg_edge_index = edge_idxs.clone().repeat(eta, 1)
    neg_edge_index[mask_1, 0] = torch.randint(num_nodes, (1, mask_1.sum()), device=DEVICE)
    neg_edge_index[mask_2, 1] = torch.randint(num_nodes, (1, mask_2.sum()), device=DEVICE)

    return neg_edge_index


def train_standard_lp(config, model_lp):
    """
    Train a standard link prediction model.

    :param config: dictionary with configuration parameters (dataset, batch_size, num_epochs, learning_rate,...)
    :param model_lp: PyTorch model for link prediction
    :return: Nothing
    """
    model_lp.train()
    start = time.time()

    dataset = config['dataset']

    loss_function_model = torch.nn.BCELoss(reduction='mean')
    if config['alpha'] > 0:
        optimizer = torch.optim.Adam(list(model_lp.parameters()), lr=config['lr'])
    else:
        optimizer = torch.optim.Adam(model_lp.parameters(), lr=config['lr'])

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
def compute_rank(out):
    # The first element of the output is the true triple score
    true_score = out[0]
    # The other elements are the corrupted triple scores
    corrupted_scores = out[1:]
    # The rank is the number of corrupted triplets that have a score higher than the true triple score
    # +1 because the true triple itself is counted -> the highest possible rank is 1
    rank = (corrupted_scores > true_score).sum().item() + 1
    return rank


@torch.no_grad()
def compute_mrr_triple_scoring(model_lp, dataset, eval_edge_index, eval_edge_type, fast=False):
    model_lp.eval()
    ranks = []
    num_samples = eval_edge_type.numel() if not fast else 5000

    # Iterate over all triples to be scored
    for triple_index in tqdm(range(num_samples)):
        # Get the triple (src, rel, dst)
        (src, dst), rel = eval_edge_index[:, triple_index], eval_edge_type[triple_index]

        # TODO: Is this not the same as negative_sampling?

        # HEAD PREDICTION TASK

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(dataset.num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (dataset.edge_index_train, dataset.edge_type_train),
            (dataset.edge_index_val, dataset.edge_type_val),
            (dataset.edge_index_test, dataset.edge_type_test),
        ]:
            # Set the mask for all true triplets to false
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        # Select all nodes that are not the true tail
        tail = torch.arange(dataset.num_entities)[tail_mask]
        # Add the true tail to the front of the list of tail nodes to be scored
        tail = torch.cat([torch.tensor([dst]), tail])
        # Create a list with src as a value for all tail nodes
        head = torch.full_like(tail, fill_value=src)
        # Create a list with rel as a value for all tail nodes
        eval_edge_typ_tensor = torch.full_like(tail, fill_value=rel).to(DEVICE)

        # Score all triples (one true triple and all corrupted triples)
        out = model_lp.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        # Compute the rank of the true triple
        rank = compute_rank(out)
        # Add the rank to the list of ranks
        ranks.append(rank)

        # TAIL PREDICTION TASK

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(dataset.num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (dataset.edge_index_train, dataset.edge_type_train),
            (dataset.edge_index_val, dataset.edge_type_val),
            (dataset.edge_index_test, dataset.edge_type_test),
        ]:
            # Set the mask for all true triplets to false
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        # Select all nodes that are not the true head
        head = torch.arange(dataset.num_entities)[head_mask]
        # Add the true head to the front of the list of head nodes to be scored
        head = torch.cat([torch.tensor([src]), head])
        # Create a list with dst as a value for all head nodes
        tail = torch.full_like(head, fill_value=dst)
        # Create a list with rel as a value for all head nodes
        eval_edge_typ_tensor = torch.full_like(head, fill_value=rel).to(DEVICE)

        # Score all triples (one true triple and all corrupted triples)
        out = model_lp.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        # Compute the rank of the true triple
        rank = compute_rank(out)
        # Add the rank to the list of ranks
        ranks.append(rank)

    # Convert the list of ranks to a tensor
    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)

    # Compute metrics
    mr = ranks.mean().item()
    mrr = (1. / ranks).mean().item()
    hits_at_10 = (ranks[ranks <= 10].size(0) / num_ranks)
    hits_at_5 = (ranks[ranks <= 5].size(0) / num_ranks)
    hits_at_3 = (ranks[ranks <= 3].size(0) / num_ranks)
    hits_at_1 = (ranks[ranks <= 1].size(0) / num_ranks)

    return mr, mrr, hits_at_10, hits_at_5, hits_at_3, hits_at_1


def train_lp_objective(config, model_lp):
    dataset = config['dataset']

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
        train_standard_lp(config, model_lp)
        # Evaluating
        if epoch % config['val_every'] == 0:
            print("Evaluating model...")
            # TODO: Ggf nicht nur auf 5000 sondern auf allen Tripeln evaluieren
            mr, mrr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model_lp,
                                                                              dataset,
                                                                              dataset.edge_index_val,
                                                                              dataset.edge_type_val,
                                                                              fast=True)
            print('val mr:', mr, 'mrr:', mrr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

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
    mr, mrr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model_lp,
                                                                      dataset,
                                                                      dataset.edge_index_test,
                                                                      dataset.edge_type_test,
                                                                      fast=True)
    print('test mr:', mr, 'mrr:', mrr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

    # Save test results
    with open(f"results/{RUN_NAME}_test_results.txt", "w") as f:
        f.write(f"test mr: {mr}, mrr: {mrr}, hits@10: {hits10}, hits@5: {hits5}, hits@3: {hits3}, hits@1: {hits1}")

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
