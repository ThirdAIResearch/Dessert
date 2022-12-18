import numpy as np
from tqdm import tqdm
num_embeddings = 600000000
buffer = np.zeros((num_embeddings, 128), dtype="float32")

FOLDER = "/scratch/jae4/msmarco"

def load(i):
    return np.load(f"{FOLDER}/encodings{i}_float32.npy")

current_start = 0
for i in tqdm(range(354)):
    embeddings = load(i)
    current_end = current_start + len(embeddings)
    buffer[current_start: current_end] = embeddings
    current_start = current_end

print(current_end)
np.save(f"{FOLDER}/all_embeddings_float32.npy", buffer[:current_end])