import numpy as np
import torch
import time
from tqdm import tqdm
import os

xmult_size = 100

start_time = time.time()
for chunk_id in tqdm(range(0, 6000)):
    all_centroid_ids = []
    
    next_chunk_filename = f"/scratch/jae4/msmarco-v2/embeddings/encoding{chunk_id}_float16.npy"
    while not os.path.exists(next_chunk_filename):
        time.sleep(5000)
    chunk_embeddings = np.load(next_chunk_filename)

    num_embeddings = len(chunk_embeddings)
    chunk_embeddings = torch.from_numpy(chunk_embeddings).to("cuda")

    for chunk_start in range(0, num_embeddings, mult_size):
        result = torch.argmax(chunk_embeddings[chunk_start: chunk_start + mult_size] @ centroids, dim=1)
        all_centroid_ids += result.to("cpu").tolist()

    np.save(f"/scratch/jae4/msmarco-v2/centroids/centroids{chunk_id}.npy", np.array(all_centroid_ids, dtype="int32"))


end_time = time.time()
torch.cuda.synchronize()

print(end_time - start_time)

