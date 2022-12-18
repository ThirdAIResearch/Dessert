import numpy as np
import torch
import time

centroids = torch.from_numpy(np.load("/scratch/jae4/msmarco/centroids.npy").astype("float16")).t().to("cuda")
mult_size = 100

start_time = time.time()
for chunk_id in range(50):
    all_centroid_ids = []
    chunk_embeddings = np.load(f"/scratch/jae4/msmarco-v2/embeddings/encoding{chunk_id}_float16.npy")
    num_embeddings = len(chunk_embeddings)
    chunk_embeddings = torch.from_numpy(chunk_embeddings).to("cuda")

    for chunk_start in range(0, num_embeddings, mult_size):
        result = torch.argmax(chunk_embeddings[chunk_start: chunk_start + mult_size] @ centroids, dim=1)
        all_centroid_ids += result.to("cpu").tolist()

    np.save(f"/scratch/jae4/msmarco-v2/centroids/centroids{chunk_id}.npy", np.array(all_centroid_ids, dtype="int32"))


end_time = time.time()
torch.cuda.synchronize()

print(len(all_centroid_ids), end_time - start_time)

