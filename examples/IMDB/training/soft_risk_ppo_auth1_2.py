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

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, set_seed ,PPOTrainer
#from trl.trainer.ppo_trainer_2 import PPOTrainer
from trl.core import LengthSampler


tqdm.pandas()

import datetime
import os
from transformers import GPT2Tokenizer, GPT2Model
import pdb
import pandas as pd
import math 

date_n_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging_dir = f"/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/imdb_final_1/sft_p_new_schedule/{date_n_time}"
os.makedirs(logging_dir)

# path to the base model 
# npn: /mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/sft_both/pn/checkpoint-200
# p: /mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/sft_final/2024-01-24_21-17-41/checkpoint-100

@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/sft_final/2024-01-24_21-17-41/checkpoint-100", #"lvwerra/gpt2-imdb",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1.41e-5,
            #log_with=None,
            mini_batch_size=128,
            batch_size=128,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
            log_with="tensorboard",
            project_kwargs={"logging_dir": logging_dir},
            steps= 128000,
            exp_name="ppo"
        )
    )
    query_dataset: str = field(default="imdb", metadata={"help": "the dataset to query"})
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
    input_query_size: str = field(default="less", metadata={"help": "the dataset to query"})
    
    generation_kwargs_min_length: Optional[int] = field(default=-1, metadata={"help": "minimum number of tokens while generation"})
    generation_kwargs_top_k: Optional[int] = field(default=0, metadata={"help": "gneration top k"})
    generation_kwargs_top_p: Optional[float] = field(default=1.0, metadata={"help": "gneration top p"})
    generation_kwargs_max_new_tokens: Optional[int] = field(default=32, metadata={"help": "gneration top p"})

    change_tokenizer_args: bool = field(default=False, metadata={"help": "whether to use modify tokenizer settings"})
    tokenizer_kwargs: dict = field(
    default_factory=lambda: {
      "padding_side": "left", 
      "truncation_side": "left",
      "pad_token_as_eos_token": True,
      "max_length": 64
    }
  )
    
    risk_scheduler: str = field(default="old", metadata={"help": "old risk scheduler that doesn't go down to alpha"})
    risk_n: int = field(default=240, metadata={"help": "240 (> batch size): no RA; 70: RA begins after iter 70; 1: RA throughout"})
    risk_alpha: Optional[float] = field(default=0.2, metadata={"help": "risk alpha value = 20 percent"})
    risk_rho: Optional[float] = field(default=0.8, metadata={"help": "risk alpha reaches a value = 20 percent at 80 percent of total iterations"})

args = tyro.cli(ScriptArguments)

generation_kwargs = {
      "min_length": args.generation_kwargs_min_length, 
      "top_k": args.generation_kwargs_top_k,
      "top_p": args.generation_kwargs_top_p,
      "do_sample": True,
      "max_new_tokens": args.generation_kwargs_max_new_tokens
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
    # load imdb with datasets: for reducing data size, just use +'[:200]'
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)


    if args.input_query_size == "more":
        def input_size():
            return 64 
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

#pdb.set_trace()

if args.change_tokenizer_args == True: 
    tokenizer.padding_side = args.tokenizer_kwargs['padding_side']
    tokenizer.truncation_side = args.tokenizer_kwargs['truncation_side']
    tokenizer.pad_token_as_eos_token = args.tokenizer_kwargs['pad_token_as_eos_token']
    tokenizer.max_length = args.tokenizer_kwargs['max_length']

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, model_name = args.ppo_config.reward_model.split(":")
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
generation_kwargs['pad_token_id'] = tokenizer.eos_token_id

# empty list to store text data
text_train_data = []

def get_current_risk_level(alpha, rho, m, M):
    """
    Get current risk_level_alpha*N value

    Args:
        alpha (float):
            Risk level alpha 
        rho (float):
            Risk level scheduler: 0.8 means the soft risk level reaches α after ρ = 80% of the training
        m (int):
            current training step/epoch
        M (int):
            total policy training steps

    Returns:
        alpha_N (int):
            risk_level_alpha*N value
    """
    alpha = args.risk_alpha
    rho = args.risk_rho

    if args.risk_scheduler == "old": 
        if m <= args.risk_n:
            return args.ppo_config.batch_size
        else:
            return math.ceil(args.ppo_config.batch_size * max(alpha, 1 - (1 - alpha) * (m - args.risk_n) / (rho * M)))   
    else: 
        print('here')
        if m <= args.risk_n:
            val =  args.ppo_config.batch_size
        elif m >= math.ceil(rho*M):
            val = math.ceil(alpha*args.ppo_config.batch_size)
        else:
            K = (1 - alpha)/(math.ceil(rho*M)-args.risk_n)
            val = math.ceil(args.ppo_config.batch_size * max(alpha, 1 - K * (m - args.risk_n)))
        return val



def get_current_risk_level_2(alpha, rho, m, M):
    """
    Get current risk_level_alpha*N value

    Args:
        alpha (float):
            Risk level alpha 
        rho (float):
            Risk level scheduler: 0.8 means the soft risk level reaches α after ρ = 80% of the training
        m (int):
            current training step/epoch
        M (int):
            total policy training steps

    Returns:
        alpha_N (int):
            risk_level_alpha*N value
    """
    alpha = 0.2
    rho = 0.8 
    n = 70
    M = 194
    
    if m <= n:
        val =  args.ppo_config.batch_size
    elif m >= math.ceil(rho*M):
        val = alpha*100
    else:
        K = (1 - alpha)/(math.ceil(rho*M)-n)
        val = math.ceil(args.ppo_config.batch_size * max(alpha, 1 - K * (m - n)))
    return val

def apply_risk_modification_to_batch(batch, query_tensors, response_tensors, rewards, alpha_N):
    query_tensors = list(map(lambda t: t.cpu(), query_tensors))
    response_tensors = list(map(lambda t: t.cpu(), response_tensors))
    rewards = list(map(lambda t: t.cpu(), rewards))
    #response_lengths = [response_tensors[i].shape[0] for i in response_tensors]

    step_dict = {
        'query_t' : query_tensors,
        'resp_t' : response_tensors,
        'rewards' : rewards
    }

    #batch = {k: [t.cpu() for t in v] if isinstance(v, list) else v.cpu() for k, v in batch.items()}
    batch = {k: [t.cpu() for t in v] if isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v) else v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    #pdb.set_trace()

    df_step_dict = pd.DataFrame(step_dict)
    df_batch = pd.DataFrame(batch)

    df_step_dict = df_step_dict.sort_values(by='rewards')#.head(alpha_N)
    df_batch = df_batch.sort_values(by='rewards')#.head(alpha_N).drop('rewards', axis=1)
    batch = df_batch.to_dict(orient='list')  
    # line #996 in ppo_trainer already does response_masks_batch[j] = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]
    #pdb.set_trace()
    #mask = [torch.cat((torch.ones_like(t, dtype =int), torch.zeros([48 - t.shape[0]], dtype =int))) for t in response_tensors]  #torch.zeros([64], dtype =int), 
    mask = [torch.ones([48], dtype =int) for _ in response_tensors]  #torch.zeros([64], dtype =int), 
    for ii in range(args.ppo_config.batch_size - alpha_N):
        mask[-ii-1] = torch.zeros([48], dtype=int)

    sr_query_tensors = list(map(lambda t: t.to(device), df_step_dict['query_t'].tolist()))
    sr_response_tensors = list(map(lambda t: t.to(device), df_step_dict['resp_t'].tolist()))
    sr_rewards = list(map(lambda t: t.to(device), df_step_dict['rewards'].tolist()))
    batch = {k: [t.to(device) for t in v] if isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v) else v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    return batch, sr_query_tensors, sr_response_tensors, sr_rewards, mask


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

    if args.reward_function == "trl":
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    else: 
        rewards = [
            torch.nn.functional.softmax(torch.tensor([output[0]["score"], output[1]["score"]]))[1]
            for output in pipe_outputs
        ]
    
    for ii in range(3): 
        data_row = {'iter': epoch, 'input':batch["query"][ii], 'output':batch["response"][ii], 'score':rewards[ii].item()}
        text_train_data.append(data_row)
    
    batch["rewards"] = rewards
    alpha_N_ceil = get_current_risk_level(PPOConfig.soft_risk_alpha, PPOConfig.risk_level_scheduler, epoch, 194)
    #sr_batch, sr_query_t, sr_resp_t, sr_rewards, mask = apply_risk_modification_to_batch(batch, query_tensors, response_tensors, rewards, alpha_N_ceil)

    # Run PPO step
    print(alpha_N_ceil)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards, alpha_N_ceil)
    ppo_trainer.log_stats(stats, batch, rewards)
    print(epoch)

ppo_trainer._save_pretrained(logging_dir)
df = pd.DataFrame(text_train_data)
df.to_csv(logging_dir+'/textual_data.csv', index=False)