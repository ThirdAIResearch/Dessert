from tqdm import tqdm
import numpy as np
import torch
import sys
import time
from ms_marco_eval import compute_metrics_from_files

sys.path.insert(0, '/home/josh/ColBERT')
from colbert import Searcher

FOLDER = "/share/josh/msmarco"
result_file_name = f"colbert.tsv"

searcher = Searcher(index="/share/josh/dessert/default_settings")

for k in [10, 1000]:
    for i in range(3):
        print(f"ColBERT trial {i}, k = {k}")

        all_pids = []
        times = []
        query_embeddings = np.load(f"{FOLDER}/small_queries_embeddings.npy")
        query_len = 32
        for query_id in tqdm(list(range(len(query_embeddings) // query_len))):
            embeddings = query_embeddings[query_id * query_len : (query_id + 1) * query_len]
            start = time.time()
            torch_embeddings = torch.unsqueeze(torch.from_numpy(embeddings), 0)
            pids, _, _ = searcher.dense_search(torch_embeddings, k=k)
            took = time.time() - start

            all_pids.append(pids)
            times.append(took)

        print("#####################")
        print("Query time metrics:")
        print(
            "P50:",
            np.percentile(times, 50),
            "P95:",
            np.percentile(times, 95),
            "P99:",
            np.percentile(times, 99),
        )
        print("Mean: ", sum(times) / len(times))
        print("#####################")

        with open(f"{FOLDER}/queries.dev.small.tsv") as f:
            qid_map = [int(line.split()[0]) for line in f.readlines()]

        with open(result_file_name, "w") as f:
            for qid_index, r in enumerate(all_pids):
                for rank, pid in enumerate(r):
                    qid = qid_map[qid_index]
                    f.write(f"{qid}\t{pid}\t{rank + 1}\n")

        metrics = compute_metrics_from_files(
            "/share/josh/msmarco/qrels.dev.small.tsv", result_file_name
        )
        print("#####################")
        for metric in sorted(metrics):
            print("{}: {}".format(metric, metrics[metric]))
        print("#####################", flush=True)