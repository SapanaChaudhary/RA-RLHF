import argparse
import csv
import pandas as pd
import evaluate
import numpy as np
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# from trl.import_utils import is_npu_available, is_xpu_available

prompt_len = 8
toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")


def build_dataset():
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/sft_auth2/2024-01-31_18-06-18"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    df = pd.read_csv(
        "/mnt/research/Anon2/Students/auth2/repos/trl/examples/Jigsaw/dataset/jigsaw_test.csv",
    )
    # filter out toxic rows
    df = df[df["toxic"] == 1]
    ds = Dataset.from_pandas(df)
    ds = ds.rename_columns({"comment_text": "review"})

    # pdb.set_trace()

    def input_size():
        return prompt_len

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# get test dataset
ds = build_dataset()


# ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test")

parser = argparse.ArgumentParser(description="Evaluate de-toxified models")
parser.add_argument("--model_type", default="all", type=str, help="Relative path to the source model folder")
parser.add_argument(
    "--output_file", default="toxicity_6B.csv", type=str, help="Relative path to the source model folder"
)
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--num_samples", default=400, type=int, help="Number of samples")
parser.add_argument("--context_length", default=2000, type=int, help="Number of samples")
parser.add_argument("--max_new_tokens", default=32, type=int, help="Max new tokens for generation")
args = parser.parse_args()


if args.model_type == "all":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detoxs",
    ]
elif args.model_type == "gpt-neo":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
    ]
elif args.model_type == "gpt-j":
    MODELS_TO_TEST = [
        # "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detox",
    ]
else:
    MODELS_TO_TEST = [args.model_type]
NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
output_file = args.output_file
max_new_tokens = args.max_new_tokens
context_length = args.context_length
# if is_xpu_available():
#     device = torch.xpu.current_device()
# elif is_npu_available():
#     device = torch.npu.current_device()
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# consider only toxic prompts
# ds = ds.filter(lambda x: x["label"] == 1)

toxicities = {}

# open a csv file
file = open(f"{output_file}", "w", newline="")
writer = csv.writer(file)
# add first rows
writer.writerow(["model_id", "mean_toxicity", "std_toxicity"])


for model_id in tqdm(MODELS_TO_TEST):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    input_texts = []

    for i, example in enumerate(ds):
        # set seed
        torch.manual_seed(42)

        input_text = example["review"]
        input_texts.append(input_text[:2000])

        if i > NUM_SAMPLES:
            break

        if (i + 1) % BATCH_SIZE == 0:
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
            inputs.input_ids = inputs.input_ids[:context_length]
            inputs.attention_mask = inputs.attention_mask[:context_length]
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [
                generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
            ]
            toxicity_score = toxicity.compute(predictions=generated_texts)
            input_texts = []

            if model_id not in toxicities:
                toxicities[model_id] = []
            toxicities[model_id].extend(toxicity_score["toxicity"])

    # last batch
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=30)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
    toxicity_score = toxicity.compute(predictions=generated_texts)
    toxicities[model_id].extend(toxicity_score["toxicity"])

    # compute mean & std using np
    mean = np.mean(toxicities[model_id])
    std = np.std(toxicities[model_id])

    # save to file
    writer.writerow([model_id, mean, std])

    # print
    print(f"Model: {model_id} - Mean: {mean} - Std: {std}")

    model = None
    # if is_xpu_available():
    #     torch.xpu.empty_cache()
    # elif is_npu_available():
    #     torch.npu.empty_cache()
    # else:
    torch.cuda.empty_cache()

# close file
file.close()
