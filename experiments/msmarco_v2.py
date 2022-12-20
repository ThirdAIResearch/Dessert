import dessert_py
import numpy as np 
from tqdm import tqdm
import time
import torch
import os

gt = []
with open("/scratch/jae4/msmarco-v2/passv2_dev_gts") as f:
    gt = [int(i.strip()) for i in f.readlines()]

query_embeddings = np.load("/scratch/jae4/msmarco-v2/passv2_dev_query_embeddings.npy")
query_len = 32

centroids = np.load("/scratch/jae4/msmarco/centroids.npy")

num_tables = 32
hashes_per_table = 5
initial_filter_k = 32768
top_k_return = 1000
nprobe_query = 2

serialize_file_name = (
    f"/scratch/jae4/msmarco-v2/indices/serialized_msmarco_v2_dessert_{hashes_per_table}_{num_tables}.index"
)

if not os.path.exists(serialize_file_name):

    index = dessert_py.DocRetrieval(
        dense_input_dimension=128,
        num_tables=num_tables,
        hashes_per_table=hashes_per_table,
        centroids=centroids,
    )

    doc_id = 0
    for chunk_idx in tqdm(range(3000)):
        assert doc_id % 25000 == 0

        centroid_ids = np.load(f"/scratch/jae4/msmarco-v2/centroids/centroids{chunk_idx}.npy")
        doclens = np.load(f"/scratch/jae4/msmarco-v2/embeddings/doclens{chunk_idx}.npy")
        embeddings = np.load(f"/scratch/jae4/msmarco-v2/embeddings/encoding{chunk_idx}_float16.npy").astype("float32")
        
        current_start = 0
        for doclen in doclens:
            current_end = current_start + doclen
            index.add_doc(
                doc_id=str(doc_id),
                doc_embeddings=embeddings[current_start: current_end],
                doc_centroid_ids=centroid_ids[current_start: current_end]
            )
            doc_id += 1
            current_start = current_end
        assert current_end == sum(doclens)

    index.serialize_to_file(serialize_file_name)

index = dessert_py.DocRetrieval.deserialize_from_file(serialize_file_name)


centroids = torch.from_numpy(centroids.transpose())

all_pids = []
times = []
recalled_1000 = 0
recalled_100 = 0
rr_100 = 0
for query_id in tqdm(range(len(query_embeddings) // query_len)):
    embeddings = query_embeddings[query_id * query_len : (query_id + 1) * query_len]
    start = time.time()

    torch_embeddings = torch.from_numpy(embeddings)

    if nprobe_query == 1:
        centroid_ids = torch.argmax(
            torch_embeddings @ centroids, dim=1
        ).tolist()
    else:
        centroid_ids = (
            torch.topk(torch_embeddings @ centroids, nprobe_query, dim=1)
            .indices.flatten()
            .tolist()
        )

    results = index.query(
        embeddings,
        num_to_rerank=initial_filter_k,
        top_k=top_k_return,
        query_centroid_ids=centroid_ids,
    )

    took = time.time() - start

    all_pids.append([int(r) for r in results])
    times.append(took)

    if gt[query_id] in all_pids[-1][:100]:
        recalled_100 += 1
        rr_100 += 1 / (1 + all_pids[-1][:100].index(gt[query_id]))

    if gt[query_id] in all_pids[-1][:1000]:
        recalled_1000 += 1

num_queries = (len(query_embeddings) // query_len)
print(recalled_100 / num_queries, recalled_1000 / num_queries, rr_100 / num_queries)