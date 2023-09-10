# LiteralE with Feature Variations

This code is the implementation of the LiteralE method from the LiteralE paper alongside the feature variations tested in my Bachelor's thesis.

## Installation

To install the required packages, run

```bash
pip3 install -r requirements.txt
```

## Usage

### Base model
To reproduce the results from the thesis for the base DistMult model on FB15k-237 with base settings, run the following command:

```bash
python3 main.py
```

### LiteralE models with Feature Variations
For the LiteralE model run

```bash
python3 main.py --lit_mode [num|txt|all|attr]
```

where ```num``` gives the LiteralE model only numerical literal features,
```txt``` only textual literal features, ```all``` both numerical and textual literal features,
and ```attr``` only the numerical literals' attributive relation types as features.
The latter corresponds to the second feature variation of the thesis, whereas the three other literal modes correspond to its first feature variation.

The third feature variation (frequency-based curated literal features) can be run by adding the ```--filter``` flag
and specifying a frequency threshold as an integer behind it.
For example, the command

```bash
python3 main.py --lit_mode num --filter 20
```

reproduces one of the experiments in the thesis for the third feature variation by filtering out all
literals with attributive relation types that occur at most 20 times.

The fourth feature variation (features of clustered textual literal embeddings) can be run by adding the ```--cluster``` flag
and specifying the number of clusters as an integer behind it.
For example, the command

```bash
python3 main.py --lit_mode txt --cluster 100
```

reproduces one of the experiments in the thesis for the fourth feature variation by clustering the textual literal embeddings
into 100 clusters.

### Other settings

To run the models on YAGO3-10, add the ```--dataset YAGO3-10``` flag.

To run the models with the ComplEx scoring function, add the ```--scoring ComplEx``` flag.

Other hyperparameters tested in this thesis that can be modified are
* ```--epochs``` (number of epochs as an integer, default 1000),
* ```--eta``` (number of negative samples per positive sample, deafult 200),
* ```--emb_dim``` (number of dimensions of the embedding space for entities and relations, default 200),
* ```--reg``` (regularization weight, default 0.0).


