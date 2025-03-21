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
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, set_seed
from trl.core import LengthSampler
from trl.trainer.ppo_trainer_original import PPOTrainer

tqdm.pandas()

import datetime
import os
from transformers import GPT2Tokenizer, GPT2Model
import pdb
import pandas as pd

date_n_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging_dir = f"/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/OLID/{date_n_time}"
os.makedirs(logging_dir)


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            # model_name="/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/sft_auth2/2024-01-20_20-32-35",  # sft_both
            model_name="/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/sft_auth2/OLID/2024-02-18_15-25-28",  # OLID sft_positive
            query_dataset="olid",
            reward_model="pigeon-phobia/bertweet-base_finetuned_olid_a",
            # reward_model="citizenlab/twitter-xlm-roberta-base-sentiment-finetunned",
            learning_rate=1.41e-5,
            # log_with=None,
            mini_batch_size=128,
            batch_size=128,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            # adap_kl_ctrl=False,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
            log_with="tensorboard",
            project_kwargs={"logging_dir": logging_dir},
            steps=192000,
            exp_name="ppo",
        )
    )
    query_dataset: str = field(default="olid", metadata={"help": "the dataset to query"})
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq models"})
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    reward_function: str = field(default="trl", metadata={"help": "whether to use trl or rl4lm reward"})
    input_query_size: str = field(default="more", metadata={"help": "the dataset to query"})

    generation_kwargs_min_length: Optional[int] = field(
        default=32, metadata={"help": "minimum number of tokens while generation"}
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
    prompt_len: Optional[int] = field(default=32, metadata={"help": "prompt length"})


args = tyro.cli(ScriptArguments)


generation_kwargs = {
    "min_length": args.generation_kwargs_min_length,
    "top_k": args.generation_kwargs_top_k,
    "top_p": args.generation_kwargs_top_p,
    "do_sample": True,
    "max_new_tokens": args.generation_kwargs_max_new_tokens,
}

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
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
    # # Kaggle datasets have to be downloaded manually (using Kaggle API)
    # # datasets.load_dataset("jigsaw_toxicity_pred", data_dir="<path/to/manual/data>")
    # ds = load_dataset(
    #     query_dataset,
    #     data_dir="/mnt/research/Anon2/Students/auth2/repos/trl/examples/Jigsaw/dataset/original_dataset",
    #     split="train",
    # )  # Hardcoding path for now TODO: change this

    # ds = ds.rename_columns({"comment_text": "review"})
    # ds = ds.filter(lambda x: len(x["review"]) > 64, batched=False)

    # df = pd.DataFrame(ds)
    # num_toxic = df["toxic"].sum()

    # toxic_df = df[df["toxic"] == True]
    # non_toxic_df = df[df["toxic"] == False]

    # non_toxic_df = non_toxic_df.sample(n=num_toxic, random_state=config.seed)

    # # Recombine into dataset
    # ds = ds.from_pandas(pd.concat([toxic_df, non_toxic_df], ignore_index=True))
    df = pd.read_csv(
        "/mnt/research/Anon2/Students/auth2/repos/trl/examples/OLID/dataset/OLIDv1.0/olid-train.csv",
    )
    ds = Dataset.from_pandas(df)
    ds = ds.rename_columns({"tweet": "review"})

    # pdb.set_trace()

    if args.input_query_size == "more":

        def input_size():
            return args.prompt_len

    else:
        input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(args.ppo_config, args.query_dataset)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name, trust_remote_code=True)
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    args.ppo_config.model_name,
    trust_remote_code=True,
    device_map=device_map,
    peft_config=peft_config,
)


tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# pdb.set_trace()

if args.change_tokenizer_args == True:
    tokenizer.padding_side = args.tokenizer_kwargs["padding_side"]
    tokenizer.truncation_side = args.tokenizer_kwargs["truncation_side"]
    tokenizer.pad_token_as_eos_token = args.tokenizer_kwargs["pad_token_as_eos_token"]
    tokenizer.max_length = args.tokenizer_kwargs["max_length"]

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
# task, model_name = args.ppo_config.reward_model.split(":")
task = "text-classification"
model_name = args.ppo_config.reward_model
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=model_name, device=device)
else:
    sentiment_pipe = pipeline(task, model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

# empty list to store text data
text_train_data = []

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(
        texts,
        **sent_kwargs,
        truncation=True,
    )

    if args.reward_function == "trl":
        # pdb.set_trace()
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    else:  # depreciated for jigsaw task
        raise NotImplementedError
        rewards = [
            -torch.nn.functional.softmax(torch.tensor([output[0]["score"], output[1]["score"], output[2]["score"]]))[2]
            for output in pipe_outputs
        ]
    # pdb.set_trace()
    for ii in range(3):
        data_row = {
            "iter": epoch,
            "input": batch["query"][ii],
            "output": batch["response"][ii],
            "score": rewards[ii].item(),
        }
        text_train_data.append(data_row)

    # pdb.set_trace()

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    print(epoch)

ppo_trainer._save_pretrained(logging_dir + "/save_pretrained")
df = pd.DataFrame(text_train_data)
df.to_csv(logging_dir + "/textual_data.csv", index=False)
