import pandas as pd 
import json
import tqdm

gts = pd.read_csv("/scratch/jae4/msmarco-v2/passv2_dev_qrels.tsv", sep="\t", header=None)
queries = pd.read_csv("/scratch/jae4/msmarco-v2/passv2_dev_queries.tsv", sep="\t", header=None)


# We'll take a slight hit in accuracy by just considering the top neighbor for now

pid_to_num = {}

with open("/scratch/jae4/msmarco-v2/msmarco_v2_passage/all_msmarco_passages") as f:
    for i, line in tqdm.tqdm(enumerate(f)):
        parsed = json.loads(line)
        pid_to_num[parsed["pid"]] = i


qid_to_pid = {}
for i in range(len(gts)):
    qid_to_pid[gts[0][i]] = pid_to_num[gts[2][i]]

final_gts = []
for qid in queries[0]:
    final_gts.append(qid_to_pid[qid])

with open("/scratch/jae4/msmarco-v2/passv2_dev_gts", "w") as f:
    for gt in final_gts:
        f.write(f"{gt}\n")