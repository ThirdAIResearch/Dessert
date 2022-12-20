import pandas as pd 
import torch
import numpy as np
from tqdm import tqdm

queries = pd.read_csv("/scratch/jae4/msmarco-v2/passv2_dev_queries.tsv", sep="\t", header=None)

from thirdai import embeddings

model = embeddings.DocSearchModel()

all_query_embeddings = []
for query in tqdm(queries[1]):
    all_query_embeddings.append(model.encodeQuery(query))

all_query_embeddings = torch.cat(all_query_embeddings)

np.save("/scratch/jae4/msmarco-v2/passv2_dev_query_embeddings.npy", all_query_embeddings.numpy())