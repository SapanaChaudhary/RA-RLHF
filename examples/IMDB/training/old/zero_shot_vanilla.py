# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl.core import LengthSampler


tqdm.pandas()

import pdb
import pickle
import datetime
import os
from transformers import GPT2Tokenizer, GPT2Model
import statistics
import random
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Union
import datasets
from datasets import Dataset

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with SFTTrainer
    """

    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    reward_model: Optional[str] = field(default="lvwerra/distilbert-imdb", metadata={"help": "model to obtain desired task score, e.g, sentiment score"})
    task_type: Optional[str] = field(default="sentiment-analysis", metadata={"help": "task type"})
    dataset_name: Optional[str] = field(
        default="imdb", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    seed: Optional[int] = field(default=0, metadata={"help": "seed"})
    
    # learning params
    learning_rate: Optional[float] = field(default=0.00001, metadata={"help": "the learning rate"}) #default=1.41e-5
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    # Using tokenizer max length=64 from RL4LLMs Table 4
    seq_length: Optional[int] = field(default=64, metadata={"help": "Input sequence length"}) #default=512
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    num_train_epochs: Optional[int] = field(default=10, metadata={"help": "the number of training epochs"}) #default=3
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})

    # model download, quantization and peft params
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})

    # logging
    output_dir: Optional[str] = field(default="/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/sft_positive/", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})

    # post processing
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8, data_split = 'train'):
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
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    # to load partial dataset, just change data split to split=data_split+'[:200]'
    ds = load_dataset(query_dataset, split=data_split)
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    def input_size():
        return 64 #LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

#what is the purpose of data collator? 
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# function to prepare dataloader as one included in ppo trainer class
def prepare_dataloader(dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None, batch_size=128):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader


# build and sample test dataset
test_dataset = build_dataset(args, args.dataset_name, data_split='test')

# Take 5k random samples
sample_size = 5000
random_indices = random.sample(range(len(test_dataset)), sample_size)
test_dataset = test_dataset.select(random_indices)

#create dataloader object
dataloader = prepare_dataloader(test_dataset, collator, batch_size=args.batch_size)

# set seed before initializing value head for deterministic eval
set_seed(args.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=None)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.pad_token_as_eos_token = True
tokenizer.max_length = 64

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
task, model_name = args.task_type, args.reward_model
sentiment_pipe = pipeline(task, model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": 48,
    "top_k": 50,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 48,
}

# generation pipeline
generator = pipeline('text-generation', model=model, **generation_kwargs)


# infer on test dataset
test_scores = []
for epoch, batch in tqdm(enumerate(dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2 model
    response_tensors = model(query_tensors, **generation_kwargs)
    response_tensors = ppo_tester.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    #rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    rewards = [
        torch.nn.functional.softmax(torch.tensor([output[0]["score"], output[1]["score"]]))[1]
        for output in pipe_outputs
    ]

    test_scores.append(sum(rewards)/len(rewards))
    print(sum(rewards)/len(rewards))

# print and save the final score  
date_n_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(f'/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/zero_shot/{date_n_time}')
final_test_score = sum(test_scores)/len(test_scores)     
test_scores_list = [t.item() for t in test_scores]
print(f'Total test sentiment-score and std is: {final_test_score.item(), statistics.stdev(test_scores_list)}')
pickle.dump(final_test_score.item(), open(f'/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/zero_shot/{date_n_time}/final_test_score', "wb"))



