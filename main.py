import argparse
import json
from datetime import datetime
import torch
import os.path as osp
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from torch.optim import lr_scheduler

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from tqdm import tqdm

from dataset import LiteralLinkPredDataset
from models import DistMult, ComplEx, ConvE


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


def train_standard_lp(config, model_lp, loss_function_model, optimizer, edge_index_batches, edge_type_batches):
    """
    Train a standard link prediction model.

    :param config: dictionary with configuration parameters (dataset, batch_size, num_epochs, learning_rate,...)
    :param model_lp: PyTorch model for link prediction
    :param loss_function_model: loss function for link prediction
    :param optimizer: optimizer for link prediction
    :param edge_index_batches: list of torch.tensor of shape (2, batch_size)
    :param edge_type_batches: list of torch.tensor of shape (batch_size)
    :return: Nothing
    """
    model_lp.train()

    dataset = config['dataset']

    batch_indices = np.arange(len(edge_index_batches))
    np.random.shuffle(batch_indices)

    loss_total = 0

    for batch_index in tqdm(batch_indices):
        edge_idxs, relation_idx = edge_index_batches[batch_index], edge_type_batches[batch_index]
        optimizer.zero_grad()

        edge_idxs_neg = negative_sampling(edge_idxs, dataset.num_entities, eta=config['eta'])

        out_pos, reg_pos = model_lp.forward(edge_idxs[:, 0], relation_idx, edge_idxs[:, 1])
        out_neg, reg_neg = model_lp.forward(edge_idxs_neg[:, 0], relation_idx.repeat(config['eta']),
                                            edge_idxs_neg[:, 1])

        out = torch.cat([out_pos, out_neg], dim=0)
        gt = torch.cat([torch.ones(len(relation_idx)), torch.zeros(len(relation_idx) * config['eta'])], dim=0).to(
            DEVICE)
        reg = (reg_pos + reg_neg) / 2

        loss = loss_function_model(out, gt)
        loss = loss + reg * config['reg_weight']

        loss_total += loss.item()
        loss.backward()
        optimizer.step()

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
        out, _ = model_lp.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

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
        out, _ = model_lp.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        # Compute the rank of the true triple
        rank = compute_rank(out)
        # Add the rank to the list of ranks
        ranks.append(rank)

    num_ranks = len(ranks)

    # Convert the list of ranks to a tensor
    ranks = torch.tensor(ranks, dtype=torch.float)

    # Compute metrics
    mr = ranks.mean().item()
    mrr = (1. / ranks).mean().item()
    hits_at_10 = (ranks[ranks <= 10].size(0) / num_ranks)
    hits_at_5 = (ranks[ranks <= 5].size(0) / num_ranks)
    hits_at_3 = (ranks[ranks <= 3].size(0) / num_ranks)
    hits_at_1 = (ranks[ranks <= 1].size(0) / num_ranks)

    return mr, mrr, hits_at_10, hits_at_5, hits_at_3, hits_at_1


def evaluate_lp_objective(model_lp, epoch, history, dataset):
    print("Evaluating model...")
    mr, mrr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model_lp,
                                                                      dataset,
                                                                      dataset.edge_index_val,
                                                                      dataset.edge_type_val,
                                                                      fast=False)
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
               "val_hits1": [],
               "train_time": []}

    loss_function_model = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model_lp.parameters(), lr=config['lr'])

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.25,
                                      total_iters=config['epochs'] - 150)

    train_edge_index_t = dataset.edge_index_train.t().to(DEVICE)
    train_edge_type = dataset.edge_type_train.to(DEVICE)

    edge_index_batches = torch.split(train_edge_index_t, config['batch_size'])
    edge_type_batches = torch.split(train_edge_type, config['batch_size'])

    # Evaluate model before training
    # evaluate_lp_objective(model_lp, 0, history, dataset)

    for epoch in range(start_epoch, config['epochs'] + 1):
        # Training
        print(f"--> Epoch {epoch}")
        start = time.time()

        train_standard_lp(config, model_lp, loss_function_model, optimizer, edge_index_batches, edge_type_batches)

        end = time.time()
        print('elapsed time:', end - start)

        if epoch == 1:
            history["train_time"].append(end - start)

        if epoch > 150:
            scheduler.step()

        # Evaluating
        if epoch % config['val_every'] == 0:
            history["train_time"].append(end - start)
            evaluate_lp_objective(model_lp, epoch, history, dataset)


if __name__ == '__main__':
    # Set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)

    if torch.cuda.is_available():
        print('Using CUDA')
        DEVICE = torch.device('cuda')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    else:
        print('Using CPU')
        DEVICE = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring", type=str, default="DistMult")
    parser.add_argument("--lit_mode", type=str, default="none") # "none", "num", "txt", "all", "attr"
    parser.add_argument("--filter", type=int, default=0)
    parser.add_argument("--cluster", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--eta", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, default="FB15k-237")
    args = parser.parse_args()

    # Set model type
    if args.lit_mode == "none":
        if args.scoring == "DistMult":
            model_type = "DistMult"
        elif args.scoring == "ComplEx":
            model_type = "ComplEx"
        elif args.scoring == "ConvE":
            model_type = "ConvE"
        else:
            raise ValueError("Invalid scoring function")
    else:
        if args.scoring == "DistMult":
            model_type = "DistMultLit"
        elif args.scoring == "ComplEx":
            model_type = "ComplExLit"
        elif args.scoring == "ConvE":
            model_type = "ConvELit"
        else:
            raise ValueError("Invalid scoring function")
    print(f"Model type: {model_type}")

    EPOCHS = args.epochs
    VAL_EVERY = args.val_every
    ETA = args.eta
    EMB_DIM = args.emb_dim
    LITERAL_MODE = args.lit_mode
    LITERAL_FILTER_THRESHOLD = args.filter
    LITERAL_TXT_CLUSTER = args.cluster
    REG = args.reg
    BATCH_SIZE = args.batch_size
    DATASET = args.dataset

    if DATASET != "FB15k-237" and DATASET != "YAGO3-10":
        raise ValueError("Invalid dataset name")

    print(f"Dataset: {DATASET}")

    if not osp.isfile(f'data/{DATASET}/processed.pt'):
        print('Process dataset...')
        dataset = LiteralLinkPredDataset(f'data/{DATASET}')
        torch.save(dataset, f'data/{DATASET}/processed.pt')
    print('Load processed dataset...')
    dataset = torch.load(f'data/{DATASET}/processed.pt')

    RUN_NAME = datetime.now().strftime("%m-%d_%H-%M-%S") + "_" + model_type + "_" + DATASET

    NOTES = "Inverse relations in train set"

    # default config
    config = {'dataset': dataset,
              'lit_mode': LITERAL_MODE,
              'filter': LITERAL_FILTER_THRESHOLD,
              'cluster': LITERAL_TXT_CLUSTER,
              'epochs': EPOCHS,
              'val_every': VAL_EVERY,
              'dim': EMB_DIM,
              'eta': ETA,
              'lr': 0.00065,
              'batch_size': BATCH_SIZE,
              'dropout': 0.2,
              'reg_weight': REG,
              'batch_norm': False,
              'notes': NOTES}

    # Write config to file
    with open(f"results/{RUN_NAME}_config.json", "w+") as f:
        json_config = config.copy()
        json_config['dataset'] = DATASET

        json.dump(json_config, f)

    if LITERAL_FILTER_THRESHOLD != 0:
        dataset.filter_literals_by_attr_relation_frequency(threshold=LITERAL_FILTER_THRESHOLD)

    if LITERAL_TXT_CLUSTER != 0:
        dataset.cluster_literals_txt(n_clusters=LITERAL_TXT_CLUSTER)

    if LITERAL_MODE == "attr":
        literal_info_num = dataset.attr_relations_num.to(DEVICE)
        literal_info_txt = dataset.attr_relations_txt.to(DEVICE)
    elif LITERAL_MODE == "num":
        literal_info_num = dataset.literals_num.to(DEVICE)
        literal_info_txt = None
    elif LITERAL_MODE == "txt":
        literal_info_num = None
        literal_info_txt = dataset.literals_txt.to(DEVICE)
    elif LITERAL_MODE == "all":
        literal_info_num = dataset.literals_num.to(DEVICE)
        literal_info_txt = dataset.literals_txt.to(DEVICE)
    else: # "none"
        literal_info_num = None
        literal_info_txt = None

    # Create model
    model_lp = None
    if model_type == "DistMult":
        model_lp = DistMult(dataset.num_entities,
                            dataset.num_relations,
                            config['dim'],
                            dropout=config['dropout'],
                            batch_norm=config['batch_norm'],
                            reg_weight=config['reg_weight'])
    elif model_type == "DistMultLit":
        model_lp = DistMult(dataset.num_entities,
                            dataset.num_relations,
                            config['dim'],
                            lit_mode=config['lit_mode'],
                            numerical_literals=literal_info_num,
                            text_literals=literal_info_txt,
                            dropout=config['dropout'],
                            batch_norm=config['batch_norm'],
                            reg_weight=config['reg_weight'])
    elif model_type == "ComplEx":
        model_lp = ComplEx(dataset.num_entities,
                           dataset.num_relations,
                           config['dim'],
                           dropout=config['dropout'],
                           batch_norm=config['batch_norm'],
                           reg_weight=config['reg_weight'])
    elif model_type == "ComplExLit":
        model_lp = ComplEx(dataset.num_entities,
                           dataset.num_relations,
                           config['dim'],
                           lit_mode=config['lit_mode'],
                           numerical_literals=literal_info_num,
                           text_literals=literal_info_txt,
                           dropout=config['dropout'],
                           batch_norm=config['batch_norm'],
                           reg_weight=config['reg_weight'])
    elif model_type == "ConvE":
        model_lp = ConvE(dataset.num_entities,
                         dataset.num_relations,
                         dropout=config['dropout'],
                         reg_weight=config['reg_weight'])
    elif model_type == "ConvELit":
        model_lp = ConvE(dataset.num_entities,
                         dataset.num_relations,
                         lit_mode=config['lit_mode'],
                         numerical_literals=literal_info_num,
                         text_literals=literal_info_txt,
                         dropout=config['dropout'],
                         reg_weight=config['reg_weight'])
    else:
        raise ValueError("Invalid model type")

    model_lp.to(DEVICE)

    # Print dataset info
    print("Number of entities and relations:")
    print(dataset.num_entities, dataset.num_relations)
    print("Number of training, validation and test triples:")
    print(dataset.edge_index_train.shape, dataset.edge_index_val.shape, dataset.edge_index_test.shape)
    print("Number of numerical and text literals:")
    print(dataset.literals_num.shape, dataset.literals_txt.shape)


    # train model
    print("Start training...")
    train_lp_objective(config, model_lp)

    # test model
    print("Start testing...")
    mr, mrr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model_lp,
                                                                      dataset,
                                                                      dataset.edge_index_test,
                                                                      dataset.edge_type_test,
                                                                      fast=False)
    print('test mr:', mr, 'mrr:', mrr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

    # Load history dictionary
    with open(f"results/{RUN_NAME}_history.json", "r") as f:
        history = json.load(f)

    # Calc average train time
    avg_train_time = np.mean(history["train_time"])

    # Save test results
    with open(f"results/{RUN_NAME}_test_results.txt", "w") as f:
        f.write(f"test mr: {mr:05.2f}\n"
                f"mrr: {mrr:05.4f}\n"
                f"hits@1: {hits1:05.4f}\n"
                f"hits@3: {hits3:05.4f}\n"
                f"hits@5: {hits5:05.4f}\n"
                f"hits@10: {hits10:05.4f}\n"
                f"avg_train_time: {avg_train_time:05.2f}")

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
                                  [mrr_history, mr_history, hits10_history, hits5_history, hits3_history,
                                   hits1_history],
                                  ["MRR", "MR", "Hits@10", "Hits@5", "Hits@3", "Hits@1"]):
        ax.plot(epochs[1:], history[1:])
        ax.set_title(f"{title} history")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
    plt.savefig(f"results/{RUN_NAME}_history.png")
