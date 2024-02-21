# Dessert
DESSERT (DESSERT Effeciently Searches Sets of Embeddings via Retrieval Tables) is a set of vector search algorithm written in C++ with python bindings.


## Installation

You can install DESSERT by cloning and ```cd```ing into this repo and then running ```pip3 install .```. You will need to have a C++ compiler installed on your machine, as well as Python 3.8+.


## Example Usage

Below is an example showing usage of DESSERT's main functionality: indexing, searching, and saving and loading the index.

```python

import dessert_py


# Create index
index = dessert_py.DocRetrieval(
    dense_input_dimension=128,
    num_tables=32,
    hashes_per_table=6,
    centroids=centroids,  # Centroids should be a 2d array of centroids for the distribution of individual vectors
)

# Add documents
for index, embeddings, centroid_ids in enumerate(docs):
    # Embeddings should be a 2d array of dense_input_dimension=128
    # centroid_ids should be a 1d array of indices that correspond to the nearest centroids to each embedding in embeddings
    index.add_doc(
        doc_id="DOC" + str(index),
        doc_embeddings=embeddings,
        doc_centroid_ids=centroid_ids,
    )

# Serialize and deserialize
index.serialize_to_file(serialize_file_name)
index = dessert_py.DocRetrieval.deserialize_from_file(serialize_file_name)


# Search
for query_embeddings, centroid_ids in enumerate(queries):
    results = index.query(
        query_embeddings,
        num_to_rerank=2000,  # How many documents to retrieve from the centroid id prefilter and then rerank with DESSERT
        top_k=100,  # How many documents to return
        query_centroid_ids=centroid_ids,
    )

```
    
## Reproducing Synthetic Experiments

You can reproduce the synthetic data experiments from our paper by running

```
cd experiments
python3 glove_synthetic.py > results.csv
python3 graph_synthetic.py
```
## Reproducing MSMarco Experiments

Reproducing the MSMarco experiments from our paper is more involved. You will need to create a folder that has all of the MSMarco ColBERT embeddings and doclens in .npy format, the ColBERT centroids in .npy format, and queries.dev.small.tsv and qrels.dev.small.tsv (which you can get directly from downloading MSMarco from the official source). To get the ColBERT embeddings, you will need to run ColBERT (you can run any of the indexing scripts from their official repository with their msmarco pretrained model) with the index function in collection_indexer.py modified as follows:


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


## Citation

If you find DESSERT useful, we'd love if you cited our paper:

```
@article{engels2024dessert,
  title={DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries},
  author={Engels, Joshua and Coleman, Benjamin and Lakshman, Vihan and Shrivastava, Anshumali},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
