from datasets import load_dataset, load_from_disk, concatenate_datasets
import pdb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

## Stage 1: Preprocess the dataset #################
# dataset = load_dataset("allenai/real-toxicity-prompts")


# def preprocess_function(examples):
#     # print(examples)
#     # pdb.set_trace()
#     return {
#         "Prompt": examples["prompt"]["text"],
#         "Continuation": examples["continuation"]["text"],
#         "Label": examples["prompt"]["toxicity"] > 0.5 if examples["prompt"]["toxicity"] else None,
#         "Perspective": examples["prompt"]["toxicity"],
#     }


# processed_dataset = (
#     dataset["train"].map(preprocess_function).select_columns(["Prompt", "Continuation", "Label", "Perspective"])
# )
# # Save the processed dataset
# processed_dataset.save_to_disk("real_toxicity.hf")


# # Stage 2: Append reward model scores #################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# real_toxicity = load_from_disk("real_toxicity.hf")
# # Load the toxicity models
# model_unitary = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device)
# tokenizer_unitary = AutoTokenizer.from_pretrained("unitary/toxic-bert")
# model_facebook = AutoModelForSequenceClassification.from_pretrained(
#     "facebook/roberta-hate-speech-dynabench-r4-target"
# ).to(device)
# tokenizer_facebook = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")


# def preprocess_function(examples):
#     # Tokenize the prompts
#     prompt_encodings_unitary = tokenizer_unitary(
#         examples["Prompt"], truncation=True, padding=True, return_tensors="pt"
#     ).to(device)
#     prompt_encodings_facebook = tokenizer_facebook(
#         examples["Prompt"], truncation=True, padding=True, return_tensors="pt"
#     ).to(device)

#     # Compute the toxicity scores
#     with torch.no_grad():
#         logits = model_unitary(**prompt_encodings_unitary).logits.float()
#         unitary_scores = (-logits[:, 1]).tolist()
#         logits = model_facebook(**prompt_encodings_facebook).logits.float()
#         facebook_scores = (logits[:, 0]).tolist()

#     # Create the labels

#     return {"Unitary": unitary_scores, "Facebook": facebook_scores}


# real_toxicity = real_toxicity.map(preprocess_function, batched=True, batch_size=1024)
# real_toxicity.save_to_disk("real_toxicity_prompts.hf")

# Stage 3: Balance dataset  and create train test split
real_toxicity = load_from_disk("real_toxicity_prompts.hf")

# Balance the dataset
real_toxicity = real_toxicity.filter(lambda x: x["Label"] is not None)
real_toxicity_pos = real_toxicity.filter(lambda x: x["Label"] == 0)
real_toxicity_neg = real_toxicity.filter(lambda x: x["Label"] == 1)

real_toxicity_neg_test = real_toxicity_neg.train_test_split(test_size=0.2, seed=42)["test"]
real_toxicity_neg_train = real_toxicity_neg.train_test_split(test_size=0.2, seed=42)["train"]

real_toxicity_pos_test = real_toxicity_pos.train_test_split(test_size=0.2, seed=42)["test"]
real_toxicity_pos_train = real_toxicity_pos.train_test_split(test_size=0.2, seed=42)["train"]

real_toxicity_train = concatenate_datasets(
    [
        real_toxicity_neg_train,
        real_toxicity_pos_train.shuffle(seed=42).select(range(int(len(real_toxicity_neg_train) * 70 / 30))),
    ]
)
real_toxicity_test = concatenate_datasets(
    [real_toxicity_neg_test, real_toxicity_pos_test.shuffle(seed=42).select(range(len(real_toxicity_neg_test)))]
)

print("Real Toxicity train size: ", len(real_toxicity_train))
print(
    "Ratio of positive to total samples: ",
    len(real_toxicity_train.filter(lambda x: x["Label"] == 0)) / len(real_toxicity_train),
)
print("Real Toxicity test size: ", len(real_toxicity_test))
print(
    "Ratio of positive to total samples: ",
    len(real_toxicity_test.filter(lambda x: x["Label"] == 0)) / len(real_toxicity_test),
)
a = input("Enter to save")
real_toxicity_train.save_to_disk(
    "/mnt/research/Anon2/Students/auth2/datasets/RealToxicity/real_toxicity_train.hf"
)
real_toxicity_test.save_to_disk(
    "/mnt/research/Anon2/Students/auth2/datasets/RealToxicity/real_toxicity_test.hf"
)
