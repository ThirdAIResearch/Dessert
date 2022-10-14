import dessert_py
import numpy as np
import time
from ms_marco_eval import compute_metrics_from_files
import torch
import os
import scann


FOLDER = "/share/josh/msmarco"

centroids = np.load(f"{FOLDER}/centroids.npy")
normalized_dataset = centroids / np.linalg.norm(centroids, axis=1)[:, np.newaxis]
searcher = (
    scann.scann_ops_pybind.builder(normalized_dataset, 1, "dot_product")
    .tree(num_leaves=1000, num_leaves_to_search=25, training_sample_size=250000)
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(25)
    .build()
)


def run_experiment(
    top_k_return,
    initial_filter_k,
    nprobe_query,
    remove_centroid_dupes,
    hashes_per_table,
    num_tables,
    use_scann=False,
):
    # ---------------------------------- Parameters --------------------------------

    serialize_file_name = (
        f"{FOLDER}/serialized_msmarco_dessert_{hashes_per_table}_{num_tables}.index"
    )

    print(
        f"""
EXPERIMENT: 
    hashes_per_table = {hashes_per_table}
    num_tables = {num_tables}
    top_k_return = {top_k_return} 
    initial_filter_k = {initial_filter_k}
    nprobe_query = {nprobe_query}
    remove_centroid_dupes = {remove_centroid_dupes},
    use_scann = {use_scann}
        """
    )

    # ----------------------- Loading Data and Building Index ----------------------

    centroids = np.load(f"{FOLDER}/centroids.npy")

    index = dessert_py.DocRetrieval(
        dense_input_dimension=128,
        num_tables=num_tables,
        hashes_per_table=hashes_per_table,
        centroids=centroids,
    )

    def get_doc_starts_and_ends(doc_lens):
        doc_starts = [0] * len(doc_lens)
        for i in range(1, len(doc_lens)):
            doc_starts[i] = doc_starts[i - 1] + doc_lens[i - 1]
        return [
            (start, start + length) for (start, length) in zip(doc_starts, doc_lens)
        ]

    def load(i):
        return np.load(f"{FOLDER}/encodings{i}_float32.npy")

    all_centroid_ids = np.load(f"{FOLDER}/all_centroid_ids.npy")
    doclens = np.load(f"{FOLDER}/doclens.npy")
    current_data_file_id = 0
    current_data_file = load(current_data_file_id)
    current_pos_in_array = 0
    doc_offsets = get_doc_starts_and_ends(doclens)

    if not os.path.exists(serialize_file_name):
        for doc_id, doc_len in enumerate(doclens):

            end_in_array = min(current_pos_in_array + doc_len, len(current_data_file))
            embeddings = current_data_file[current_pos_in_array:end_in_array]
            doc_len_left = doc_len - end_in_array + current_pos_in_array
            current_pos_in_array = end_in_array

            # Assumes doc is split over max of 2 files
            if doc_len_left > 0:
                current_data_file_id += 1
                current_data_file = load(current_data_file_id)
                end_in_array = min(
                    current_pos_in_array + doc_len, len(current_data_file)
                )
                if len(embeddings) > 0:
                    embeddings = np.concatenate(
                        embeddings, current_data_file[:doc_len_left]
                    )
                else:
                    embeddings = current_data_file[:doc_len_left]
                current_pos_in_array = doc_len_left

            index.add_doc(
                doc_id="MS" + str(doc_id),
                doc_embeddings=embeddings,
                doc_centroid_ids=all_centroid_ids[
                    doc_offsets[doc_id][0] : doc_offsets[doc_id][1]
                ],
            )

        index.serialize_to_file(serialize_file_name)

    # ---------------------------------- Evaluation --------------------------------

    index = dessert_py.DocRetrieval.deserialize_from_file(serialize_file_name)

    result_file_name = f"msmarco_{hashes_per_table}_{num_tables}_{initial_filter_k}.tsv"

    with open(f"{FOLDER}/queries.dev.small.tsv") as f:
        qid_map = [int(line.split()[0]) for line in f.readlines()]

    query_embeddings = np.load(f"{FOLDER}/small_queries_embeddings.npy")
    query_len = 32

    all_pids = []
    times = []
    centroids = torch.from_numpy(centroids.transpose())
    for query_id in range(len(query_embeddings) // query_len):
        embeddings = query_embeddings[query_id * query_len : (query_id + 1) * query_len]
        start = time.time()

        torch_embeddings = torch.from_numpy(embeddings)

        if use_scann:
            neighbors, _ = searcher.search_batched(
                embeddings, final_num_neighbors=nprobe_query
            )
            centroid_ids = neighbors.flatten().tolist()
        else:
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

        if remove_centroid_dupes:
            centroid_ids = list(set(centroid_ids))

        # print(time.time() - start)
        results = index.query(
            embeddings,
            num_to_rerank=initial_filter_k,
            top_k=top_k_return,
            query_centroid_ids=centroid_ids,
        )

        # print(time.time() - start)

        # print(time.time() - start)

        took = time.time() - start

        all_pids.append([int(r[2:]) for r in results])
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


do_hyperparameter_search = False

if do_hyperparameter_search:
    for hashes_per_table in [4, 5, 6, 7, 8]:
        for num_tables in [16, 32, 64, 128]:
            if (2**hashes_per_table) * num_tables > 2048:
                continue
            for top_k_return in [1000]:
                for initial_filter_k in [1000, 2048, 4096, 8192, 16384]:
                    if top_k_return > initial_filter_k:
                        continue
                    for nprobe_query in [1, 2, 4]:
                        run_experiment(
                            top_k_return=top_k_return,
                            initial_filter_k=initial_filter_k,
                            nprobe_query=nprobe_query,
                            remove_centroid_dupes=False,
                            hashes_per_table=hashes_per_table,
                            num_tables=num_tables,
                            use_scann=True,
                        )
else:

    for _ in range(3):
        run_experiment(
            top_k_return=1000,
            initial_filter_k=4096,
            nprobe_query=2,
            remove_centroid_dupes=False,
            hashes_per_table=7,
            num_tables=64,
            use_scann=True,
        )

    for _ in range(3):
        run_experiment(
            top_k_return=1000,
            initial_filter_k=4096,
            nprobe_query=1,
            remove_centroid_dupes=False,
            hashes_per_table=7,
            num_tables=32,
            use_scann=True,
        )

    for _ in range(3):
        run_experiment(
            top_k_return=1000,
            initial_filter_k=16384,
            nprobe_query=4,
            remove_centroid_dupes=False,
            hashes_per_table=7,
            num_tables=32,
            use_scann=True,
        )

    for _ in range(3):
        run_experiment(
            top_k_return=1000,
            initial_filter_k=8192,
            nprobe_query=4,
            remove_centroid_dupes=False,
            hashes_per_table=6,
            num_tables=32,
            use_scann=True,
        )
