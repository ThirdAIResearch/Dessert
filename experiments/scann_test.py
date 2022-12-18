import numpy as np
import scann
import os
import tqdm

from ms_marco_eval import compute_metrics_from_files

FOLDER = "/scratch/jae4/msmarco"


# all_embeddings = np.load(f"{FOLDER}/encodings0_float32.npy")

all_embeddings = np.load(f"{FOLDER}/all_embeddings_float32.npy")

doc_lens = np.load(f"{FOLDER}/doclens.npy")

if not os.path.isfile(f"{FOLDER}/eid_to_pid.npy"):
    embedding_id_to_doc_id = np.zeros(sum(doc_lens))
    current_eid = 0
    for pid, doc_len in enumerate(doc_lens):
        for _ in range(doc_len):
            embedding_id_to_doc_id[current_eid] = pid
            current_eid += 1
    np.save(f"{FOLDER}/eid_to_pid.npy", embedding_id_to_doc_id)
embedding_id_to_doc_id = np.load(f"{FOLDER}/eid_to_pid.npy")


print("STARTING BUILDING", flush=True)

searcher = (
    scann.scann_ops_pybind.builder(all_embeddings, 1, "dot_product")
    .tree(num_leaves=1000, num_leaves_to_search=25, training_sample_size=1000000)
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(25)
    .build()
)

print("ENDING BUILDING", flush=True)

with open(f"{FOLDER}/queries.dev.small.tsv") as f:
    qid_map = [int(line.split()[0]) for line in f.readlines()]

query_embeddings = np.load(f"{FOLDER}/small_queries_embeddings.npy")
query_len = 32
top_k_overall = 1000
neighbors_per_embedding = top_k_overall // query_len

all_pids = []
for query_id in tqdm.tqdm(list(range(len(query_embeddings) // query_len))):
    embeddings = query_embeddings[query_id * query_len : (query_id + 1) * query_len]
    all_neighbors, _ = searcher.search_batched(
        embeddings, final_num_neighbors=neighbors_per_embedding
    )
    all_pids.append(list({int(embedding_id_to_doc_id[eid]) for eid in all_neighbors.flatten()}))

result_file_name = "scann_results.tsv"

with open(result_file_name, "w") as f:
    for qid_index, r in enumerate(all_pids):
        for rank, pid in enumerate(r):
            qid = qid_map[qid_index]
            f.write(f"{qid}\t{pid}\t{rank + 1}\n")

metrics = compute_metrics_from_files(
    "/scratch/jae4/msmarco/qrels.dev.small.tsv", result_file_name
)
print("#####################")
for metric in sorted(metrics):
    print("{}: {}".format(metric, metrics[metric]))
print("#####################", flush=True)