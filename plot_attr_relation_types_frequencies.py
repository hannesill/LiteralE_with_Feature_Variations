import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp

if __name__ == '__main__':
    dataset_name = 'FB15k-237'
    # if not osp.isfile(f'data/{dataset_name}/processed.pt'):
    #     print('Process dataset...')
    #     dataset = LiteralLinkPredDataset(f'data/{dataset_name}')
    #     torch.save(dataset, f'data/{dataset_name}/processed.pt')
    #
    # dataset = torch.load(f'data/{dataset_name}/processed.pt')

    triple_file = f"data/{dataset_name}"

    df_literals_num = pd.read_csv(osp.join(triple_file, 'numerical_literals.txt'), header=None, sep='\t')

    # Count the number of times a literal relation occurs
    attr_relations_num_counts = df_literals_num[1].value_counts()

    print(attr_relations_num_counts)

    # Parse the literal relations to be only the text after the second last /
    attr_relations_num_counts.index = attr_relations_num_counts.index.map(lambda x: ".".join(x.split('/')[-1].split('.')[-2:]))

    print(attr_relations_num_counts)

    # Plot the counts of the literal relations
    # Show the highest x and lowest y literal relations in one horizontal bar plot
    x = 5
    y = 5
    highest_x_names = list(attr_relations_num_counts[:x].index)
    highest_x_values = list(attr_relations_num_counts[:x].values)
    lowest_y_names = list(attr_relations_num_counts[-y:].index)
    lowest_y_values = list(attr_relations_num_counts[-y:].values)

    labels = highest_x_names + ["..."] +  lowest_y_names
    labels.reverse()

    print(labels)

    values = highest_x_values + [0] + lowest_y_values
    values.reverse()

    # Plot the values logarithmically
    # Don't cut off the labels
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.4)
    plt.barh(labels, values, color="grey", log=True)
    plt.xlabel("Number of occurrences")
    # Add numbers behind the bars
    for i, v in enumerate(values):
        if v != 0:
            plt.text(v, i - 0.1, str(" " + str(v)), color='black', fontweight='bold')
        # Color the bars red if they occur more than 20 times and blue else
        if v < 20:
            plt.gca().get_children()[i].set_color('#ffeda0')
        else:
            plt.gca().get_children()[i].set_color('#f03b20')
    plt.tick_params(axis='x', which='minor', length=0)
    # Remove the top and right spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()

    # Plot all literal relations
    all_literal_relations = list(attr_relations_num_counts.index)
    all_literal_relations.reverse()
    all_literal_relations_values = list(attr_relations_num_counts.values)
    all_literal_relations_values.reverse()

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.4)
    plt.barh(all_literal_relations, all_literal_relations_values, log=True)
    plt.xlabel("Number of occurrences")
    plt.ylabel("Attributive relation types from least to most frequent")
    # Remove y ticks
    plt.yticks([])
    # Make all bars red if they occur less than 100 times and blue if they occur less than 20 times
    for i, v in enumerate(all_literal_relations_values):
        if v < 20:
            plt.gca().get_children()[i].set_color('#ffeda0')
        elif v < 100:
            plt.gca().get_children()[i].set_color('#feb24c')
        else:
            plt.gca().get_children()[i].set_color('#f03b20')
    plt.tick_params(axis='x', which='minor', length=0)
    # Remove the top and right spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()