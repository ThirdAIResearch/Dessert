import h5py
import dessert_py
import time
import torch
import torch.nn.functional as func
import numpy as np


filename = "glove-200-angular.hdf5"

num_docs = 1000
query_noise_std = 0.1

for power_of_2 in range(1, 11):
    m = 2 ** (power_of_2 + 1)
    doc_m = m
    query_m = m


    with h5py.File(filename, "r") as f:
        train = f["train"][()]

    index = dessert_py.DocRetrieval(
        dense_input_dimension=200,
        num_tables=16,
        hashes_per_table=power_of_2,
        centroids=[[0 for _ in range(200)]],
    )

    docs = []
    for doc_id in range(num_docs):
        doc = train[np.random.choice(train.shape[0], doc_m, replace=False), :]
        docs.append(doc)


    num_docs = 0
    for doc_id, doc in list(enumerate(docs)):
        num_docs += 1
        is_new = index.add_doc(
            doc_id=str(doc_id),
            doc_embeddings=doc,
        )

    def get_gt(query):
        query_torch = torch.from_numpy(query).double()
        query_torch = func.normalize(query_torch, p=2, dim=1)

        scores = []
        for doc in docs:
            result = (query_torch @ torch.from_numpy(doc).double().t())
            maxes, _ = torch.max(result, dim=1)
            scores.append((torch.sum(maxes, dim=0) / query_m).tolist())
        return np.argmax(scores)

    all_docs = torch.from_numpy(np.concatenate(docs)).t().double()
    def get_gt_fast(query):
        query_torch = torch.from_numpy(query).double()
        query_torch = func.normalize(query_torch, p=2, dim=1)

        result = (query_torch @ all_docs).t() 
        reshaped = torch.reshape(result, (num_docs, doc_m, query_m))
        maxes, _ = torch.max(reshaped, dim=1)
        scores = torch.sum(maxes, dim=1) / query_m
        return scores.argmax()



    total_our_time = 0
    total_gt_time = 0
    total_fast_gt_time = 0
    recall = 0
    for doc_id, doc in list(enumerate(docs)):
        query = doc + np.random.normal(scale=query_noise_std, size=doc.shape)
        query = query[:query_m]

        start = time.time()
        top_index = index.rerank(
            query,
            internal_ids=range(num_docs)
        )[0]
        total_our_time += time.time() - start
        recall += top_index == doc_id

        start = time.time()
        assert(get_gt_fast(query) == doc_id)
        total_fast_gt_time += time.time() - start

        start = time.time()
        assert(get_gt(query) == doc_id)
        total_gt_time += time.time() - start



    print(doc_m, query_m, recall, total_our_time / len(docs), total_gt_time / len(docs), total_fast_gt_time / len(docs), flush=True)
