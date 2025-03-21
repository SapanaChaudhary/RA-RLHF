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
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset, load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer


tqdm.pandas()

import pdb
from transformers import GenerationConfig
from transformers import AutoTokenizer, pipeline, Seq2SeqTrainingArguments
import datetime
import os
from trl import set_seed

date_n_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging_dir = (
    f"/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/sft_auth2/real_toxicity/{date_n_time}"
)
os.makedirs(logging_dir)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with SFTTrainer
    """

    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="jigsaw_toxicity_pred", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(
        default="Prompt", metadata={"help": "the text field of the dataset"}
    )
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'wandb' to log with wandb"})

    # learning params
    learning_rate: Optional[float] = field(default=0.00001, metadata={"help": "the learning rate"})  # default=1.41e-5
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    # Using tokenizer max length=64 from RL4LLMs Table 4
    seq_length: Optional[int] = field(default=64, metadata={"help": "Input sequence length"})  # default=512
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    num_train_epochs: Optional[int] = field(
        default=10, metadata={"help": "the number of training epochs"}
    )  # default=3
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
    output_dir: Optional[str] = field(default=logging_dir, metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})

    # post processing
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})

    # added by auth1
    reward_function: str = field(default="trl", metadata={"help": "whether to use trl or rl4lm reward"})
    input_query_size: str = field(default="less", metadata={"help": "the dataset to query"})
    generation_kwargs_min_length: Optional[int] = field(
        default=-1, metadata={"help": "minimum number of tokens while generation"}
    )
    generation_kwargs_top_k: Optional[int] = field(default=0, metadata={"help": "gneration top k"})
    generation_kwargs_top_p: Optional[float] = field(default=1.0, metadata={"help": "gneration top p"})
    generation_kwargs_max_new_tokens: Optional[int] = field(default=32, metadata={"help": "gneration top p"})

    change_tokenizer_args: bool = field(default=False, metadata={"help": "whether to use modify tokenizer settings"})
    tokenizer_kwargs: dict = field(
        default_factory=lambda: {
            "padding_side": "left",
            "truncation_side": "left",
            "pad_token_as_eos_token": True,
            "max_length": 64,
        }
    )
    exp_name: str = field(default="sft-positive", metadata={"help": "experiment name"})
    seed: Optional[int] = field(default=0, metadata={"help": "seed"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

generation_kwargs = {
    "min_length": script_args.generation_kwargs_min_length,
    "top_k": script_args.generation_kwargs_top_k,
    "top_p": script_args.generation_kwargs_top_p,
    "do_sample": True,
    "max_new_tokens": script_args.generation_kwargs_max_new_tokens,
}

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

# Removing huggingface authorization token param from model loading command above
#    use_auth_token=script_args.use_auth_token,
# Make sure you have logged in to the Hugging Face Hub using huggingface-cli login
# or by calling huggingface_hub.login() in Python. This will save an authentication
# token that Transformers will pick up automatically.

# Step 2: Load the dataset
dataset = load_from_disk("/mnt/shared-scratch/Shakkottai_S/auth2d36/datasets/RealToxicity/real_toxicity_train.hf")
dataset = dataset.filter(lambda x:x["Label"]==0)  # SFT only on positive(non-toxic) dataset



# Balance the dataset

# Filter dataset for only positive reviews
# dataset = dataset.filter(lambda x: x["label"] == 1, batched=False)


# Step 3: Define generation arguments
# generation_kwargs = GenerationConfig(
#     min_length = 48,
#     top_k = 50,
#     top_p = 1.0,
#     do_sample = True,
#     max_new_tokens = 48,
# )
# pad_token_id = tokenizer.eos_token_id,

# Step 3: Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    generation_config=generation_kwargs,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None


# trl sft trainer default tokenizer and pad_token
"""
if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

if max_seq_length is None:
            # to overcome some issues with broken tokenizers
            max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )

"""

# Looks like padding_side = "right" by default
# Hence, defining our tokenizer here
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token

if script_args.change_tokenizer_args == True:
    tokenizer.padding_side = script_args.tokenizer_kwargs["padding_side"]
    tokenizer.truncation_side = script_args.tokenizer_kwargs["truncation_side"]
    tokenizer.pad_token_as_eos_token = script_args.tokenizer_kwargs["pad_token_as_eos_token"]
    tokenizer.max_length = script_args.tokenizer_kwargs["max_length"]

# tokenizer.padding_side = "left"
# tokenizer.truncation_side = "left"
# tokenizer.pad_token_as_eos_token = True
# tokenizer.max_length = 64

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(logging_dir)
