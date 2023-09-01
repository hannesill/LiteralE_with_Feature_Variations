import argparse
import json

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load first argument as history file
    parser = argparse.ArgumentParser()
    parser.add_argument("history_file", help="Path to history file")
    args = parser.parse_args()
    history_file = args.history_file

    # Load history dictionary
    with open(history_file, "r") as f:
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

    # Plot all histories and save figures (exlude first epoch)
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    print(zip(axs.flatten(),
              [mrr_history, mr_history, hits10_history, hits5_history, hits3_history, hits1_history],
              ["MRR", "MR", "Hits@10", "Hits@5", "Hits@3", "Hits@1"]))
    for ax, history, title in zip(axs.flatten(),
                                  [mrr_history, mr_history, hits10_history, hits5_history, hits3_history,
                                   hits1_history],
                                  ["MRR", "MR", "Hits@10", "Hits@5", "Hits@3", "Hits@1"]):
        ax.plot(epochs[1:], history[1:])
        ax.set_title(f"{title} history")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)

    # Save figure
    RUN_NAME = history_file.split("/")[-1].split(".")[0]
    plt.savefig(f"results/{RUN_NAME}.png")
