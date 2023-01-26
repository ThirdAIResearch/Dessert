# Dessert
DESSERT (DESSERT Effeciently Searches Sets of Embeddings via Retrieval Tables) is a set of vector Search algorithm written in C++ with python bindings.


## Installation

You can install DESSERT by cloning and ```cd```ing into this repo and then running ```pip3 install .```. You will need to have a C++ compiler installed on your machine, as well as Python 3.8+.


    
## Reproducing Synthetic Experiments

You can reproduce the synthetic data experiments from our paper by running

```
cd experiments
python3 glove_synthetic.py > results.csv
python3 graph_synthetic.py
```
## Reproducing MSMarco Experiments

Reproducing the MSMarco experiments from our paper is more involved. You will need to create a folder that has all of the MSMarco ColBERT embeddings and doclens in .npy format, the ColBERT centroids in .npy format, and queries.dev.small.tsv and qrels.dev.small.tsv (which you can get directly from downloading MSMarco from the official source). To get the ColBERT embeddings, you will need to run ColBERT (from their official repository) with the index function in  modify the index function in collection_indexer.py modified as follows:


    embs, doclens = self.encoder.encode_passages(passages)

    # BEGIN INSERTED CODE:
    FOLDER = <insert your folder here>
    import numpy as np
    numpy_16 = embs.numpy().astype("float16")
    np.save(f"{FOLDER}/encoding{chunk_idx}_float16.npy", numpy_16)
    np.save(f"{FOLDER}/doclens{chunk_idx}.npy", doclens)
    # END INSERTED CODE


    if self.use_gpu:
        assert embs.dtype == torch.float16

The ColBERT centroids are automatically saved as part of indexing, so you can just convert them into .npy format and copy them into the folder.

Finally, you should change the FOLDER variable in experiments/msmarco.py to be the local folder you are using, and then you can run

```python3 msmarco.py```
