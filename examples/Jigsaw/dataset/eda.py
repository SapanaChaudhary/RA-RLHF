from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm

tqdm.pandas()
# Load IMDB dataset
# imdb_train = load_dataset("imdb", split="train")
# imdb_test = load_dataset("imdb", split="test")


# Load Jigsaw toxicity dataset
jigsaw = load_dataset(
    "OxAISH-AL-LLM/wiki_toxic",
    # data_dir="/mnt/research/Anon2/Students/auth2/repos/trl/examples/Jigsaw/dataset/original_dataset",
    # split="train",
)  # Hardcoding path for now TODO: change this


jigsaw = jigsaw.filter(lambda x: len(x["comment_text"]) > 64, batched=False)
jigsaw = jigsaw.rename_column("label", "toxic")
# imdb_train = imdb_train.filter(lambda x: len(x["text"]) > 64, batched=False)
# imdb_test = imdb_test.filter(lambda x: len(x["text"]) > 64, batched=False)


# imdb_df_train = pd.DataFrame(imdb_train)
# imdb_df_test = pd.DataFrame(imdb_test)
jigsaw_df_train = pd.DataFrame(jigsaw["train"])
jigsaw_df_test = pd.DataFrame(jigsaw["test"])


# # Balancing jigsaw dataset for toxic and non-toxic comments

num_toxic_train = jigsaw_df_train["toxic"].sum()
num_toxic_test = jigsaw_df_test["toxic"].sum()

print("Number of toxic comments in train: ", num_toxic_train)
print("Number of toxic comments in test: ", num_toxic_test)
print("Number of non-toxic comments in train: ", len(jigsaw_df_train) - num_toxic_train)
print("Number of non-toxic comments in test: ", len(jigsaw_df_test) - num_toxic_test)
a = input("Press Enter to continue...")

jigsaw_toxic_train = jigsaw_df_train[jigsaw_df_train["toxic"] == True]
jigsaw_non_toxic_train = jigsaw_df_train[jigsaw_df_train["toxic"] == False]

jigsaw_toxic_test = jigsaw_df_test[jigsaw_df_test["toxic"] == True]
jigsaw_non_toxic_test = jigsaw_df_test[jigsaw_df_test["toxic"] == False]

jigsaw_non_toxic_train = jigsaw_non_toxic_train.sample(n=num_toxic_train, random_state=42)
jigsaw_non_toxic_test = jigsaw_non_toxic_test.sample(n=num_toxic_test, random_state=42)

jigsaw_df_train = pd.concat([jigsaw_toxic_train, jigsaw_non_toxic_train], ignore_index=True)
jigsaw_df_test = pd.concat([jigsaw_toxic_test, jigsaw_non_toxic_test], ignore_index=True)

# # Save the dataset
print("Saving the dataset")
print("Jigsaw train size: ", len(jigsaw_df_train))
print("Jigsaw test size: ", len(jigsaw_df_test))
# print("IMDB train size: ", len(imdb_df_train))
# print("IMDB test size: ", len(imdb_df_test))
jigsaw_df_train.to_csv("wiki_toxic_train.csv", index=False)
jigsaw_df_test.to_csv("wiki_toxic_test.csv", index=False)

# imdb_df_train.to_csv("imdb_train.csv", index=False)
# imdb_df_test.to_csv("imdb_test.csv", index=False)


# # Load the dataset
# jigsaw_df_train = pd.read_csv("jigsaw_train.csv")
# jigsaw_df_test = pd.read_csv("jigsaw_test.csv")
# imdb_df_train = pd.read_csv("imdb_train.csv")
# imdb_df_test = pd.read_csv("imdb_test.csv")


# # Reward computation for Jigsaw dataset
# sent_kwargs = {"return_all_scores": False, "function_to_apply": "none"}
# jigsaw_reward_model = pipeline("text-classification", model="unitary/toxic-bert", truncation=True, **sent_kwargs)

# print("Calculating reward for Jigsaw train dataset")
# jigsaw_df_train["reward"] = jigsaw_df_train["comment_text"].progress_apply(
#     lambda x: -jigsaw_reward_model(x, truncation=True)[0]["score"]
# )

# print("Calculating reward for Jigsaw test dataset")
# jigsaw_df_test["reward"] = jigsaw_df_test["comment_text"].progress_apply(
#     lambda x: -jigsaw_reward_model(x, truncation=True)[0]["score"]
# )

# # Save the dataset
# print("Saving the dataset")
# print("Jigsaw train size: ", len(jigsaw_df_train))
# print("Jigsaw test size: ", len(jigsaw_df_test))
# jigsaw_df_train.to_csv("jigsaw_train.csv", index=False)
# jigsaw_df_test.to_csv("jigsaw_test.csv", index=False)


# # Reward computation for IMDB dataset
# sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
# imdb_reward_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", truncation=True, **sent_kwargs)


# print("Calculating reward for IMDB train dataset")
# imdb_df_train["reward"] = imdb_df_train["text"].progress_apply(lambda x: imdb_reward_model(x)[0][1]["score"])
# imdb_df_test["reward"] = imdb_df_test["text"].progress_apply(lambda x: imdb_reward_model(x)[0][1]["score"])

# # Save the dataset
# print("Saving the dataset")
# print("IMDB train size: ", len(imdb_df_train))
# print("IMDB test size: ", len(imdb_df_test))
# imdb_df_train.to_csv("imdb_train.csv", index=False)
# imdb_df_test.to_csv("imdb_test.csv", index=False)


# # Plot the class wise reward distribution for Jigsaw dataset

# toxic_reward = jigsaw_df_train[jigsaw_df_train["toxic"] == True]["reward"]
# non_toxic_reward = jigsaw_df_train[jigsaw_df_train["toxic"] == False]["reward"]

# plt.clf()
# plt.hist(toxic_reward, bins=100, alpha=0.5, label="Toxic")
# plt.hist(non_toxic_reward, bins=100, alpha=0.5, label="Non-Toxic")
# plt.legend(loc="upper right")
# plt.xlabel("Reward")
# plt.ylabel("Frequency")
# plt.title("Reward distribution for Jigsaw dataset (Train)")
# plt.savefig("reward_dist_jigsaw.png")

# toxic_reward = jigsaw_df_test[jigsaw_df_test["toxic"] == True]["reward"]
# non_toxic_reward = jigsaw_df_test[jigsaw_df_test["toxic"] == False]["reward"]

# plt.clf()
# plt.hist(toxic_reward, bins=100, alpha=0.5, label="Toxic")
# plt.hist(non_toxic_reward, bins=100, alpha=0.5, label="Non-Toxic")
# plt.legend(loc="upper right")
# plt.xlabel("Reward")
# plt.ylabel("Frequency")
# plt.title("Reward distribution for Jigsaw dataset (Test)")
# plt.savefig("reward_dist_jigsaw_test.png")


# # Plot the class wise reward distribution for IMDB dataset

# positive_reward = imdb_df_train[imdb_df_train["label"] == 1]["reward"]
# negative_reward = imdb_df_train[imdb_df_train["label"] == 0]["reward"]

# plt.clf()
# plt.hist(positive_reward, bins=100, alpha=0.5, label="Positive")
# plt.hist(negative_reward, bins=100, alpha=0.5, label="Negative")
# plt.legend(loc="upper right")
# plt.xlabel("Reward")
# plt.ylabel("Frequency")
# plt.title("Reward distribution for IMDB dataset (Train)")
# plt.savefig("reward_dist_imdb.png")

# positive_reward = imdb_df_test[imdb_df_test["label"] == 1]["reward"]
# negative_reward = imdb_df_test[imdb_df_test["label"] == 0]["reward"]

# plt.clf()
# plt.hist(positive_reward, bins=100, alpha=0.5, label="Positive")
# plt.hist(negative_reward, bins=100, alpha=0.5, label="Negative")
# plt.legend(loc="upper right")
# plt.xlabel("Reward")
# plt.ylabel("Frequency")
# plt.title("Reward distribution for IMDB dataset (Test)")
# plt.savefig("reward_dist_imdb_test.png")
