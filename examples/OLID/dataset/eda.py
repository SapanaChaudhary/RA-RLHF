from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
import pdb

tqdm.pandas()


# Load OLID dataset
olid = pd.read_csv(
    "/mnt/research/Anon2/Students/auth2/repos/trl/examples/OLID/dataset/OLIDv1.0/olid-training-v1.0.tsv",
    sep="\t",
)
olid_offense = olid[olid["subtask_a"] == "OFF"]
olid_not_offense = olid[olid["subtask_a"] == "NOT"]

print("Offensive: ", len(olid_offense))
print("Not Offensive: ", len(olid_not_offense))

# Load OLID test dataset
olid_test = pd.read_csv(
    "/mnt/research/Anon2/Students/auth2/repos/trl/examples/OLID/dataset/OLIDv1.0/testset-levela.tsv",
    sep="\t",
)
olid_test_labels = pd.read_csv(
    "/mnt/research/Anon2/Students/auth2/repos/trl/examples/OLID/dataset/OLIDv1.0/labels-levela.csv",
    header=None,
)
olid_test_labels.columns = ["id", "subtask_a"]
olid_test = pd.concat([olid_test, olid_test_labels["subtask_a"]], axis=1)

print("Offensive: ", len(olid_test[olid_test["subtask_a"] == "OFF"]))
print("Not Offensive: ", len(olid_test[olid_test["subtask_a"] == "NOT"]))

# save train and test datasets
olid_test.to_csv(
    "/mnt/research/Anon2/Students/auth2/repos/trl/examples/OLID/dataset/OLIDv1.0/olid-test.csv",
    index=False,
)
olid.to_csv(
    "/mnt/research/Anon2/Students/auth2/repos/trl/examples/OLID/dataset/OLIDv1.0/olid-train.csv",
    index=False,
)
