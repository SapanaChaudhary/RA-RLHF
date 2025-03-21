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
from trl.core import LengthSampler


tqdm.pandas()
import matplotlib.pyplot as plt
import itertools

import pdb
import pickle
import datetime
import os
from evaluate import load
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import random

import pandas as pd
import pdb

date_n_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(date_n_time)
logging_dir = f"/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/plot_score/{date_n_time}"
os.makedirs(logging_dir)

@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="lvwerra/gpt2-imdb",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1.41e-5,
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
            remove_unused_columns=False
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


args = tyro.cli(ScriptArguments)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

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

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# get test dataset
test_dataset = build_dataset(args.ppo_config, args.query_dataset, data_split='train')
# Take 5k random samples
# random.seed(10)
# sample_size = 2500
# random_indices = random.sample(range(len(test_dataset)), sample_size)
# test_dataset = test_dataset.select(random_indices)

#pdb.set_trace()

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

zero_shot_model = trl_model_class.from_pretrained(
    'gpt2',
    trust_remote_code=True,
    device_map=device_map,
    peft_config=peft_config,
)

sft_model = trl_model_class.from_pretrained(
    'lvwerra/gpt2-imdb',
    trust_remote_code=True,
    device_map=device_map,
    peft_config=peft_config,
)

# For the following to work I should have saved the model using huggingface
# ppo_model = trl_model_class.from_pretrained(
#     '/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/ppo/2023-09-26_11-27-23/model.pt',
#     trust_remote_code=True,
#     device_map=device_map,
#     peft_config=peft_config,
# )

sft_ppo_model = trl_model_class.from_pretrained(
    '/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/trl_sft_ppo/2023-10-23_23-30-47',
    trust_remote_code=True,
    device_map=device_map,
    peft_config=peft_config,
)
"""
ppo_model_PATH = '/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/ppo/2023-09-26_11-27-23/model.pt'
ppo_model = torch.load(ppo_model_PATH)  

sft_ppo_model_PATH = '/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/sft_ppo/2023-09-26_10-36-41/model.pt'
sft_ppo_model = torch.load(sft_ppo_model_PATH)
"""
tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.pad_token_as_eos_token = True
tokenizer.max_length = 64

# Define all the tester classes 
zero_shot_tester = PPOTrainer(args.ppo_config, zero_shot_model, ref_model, tokenizer, dataset=test_dataset, data_collator=collator)
sft_tester = PPOTrainer(args.ppo_config, sft_model, ref_model, tokenizer, dataset=test_dataset, data_collator=collator)
# ppo_tester = PPOTrainer(args.ppo_config, ppo_model, ref_model, tokenizer, dataset=test_dataset, data_collator=collator)
sft_ppo_tester = PPOTrainer(args.ppo_config, sft_ppo_model, ref_model, tokenizer, dataset=test_dataset, data_collator=collator)

#pdb.set_trace()
#ppo_tester._save_pretrained('/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/ppo/2023-09-26_11-27-23')
#sft_ppo_tester._save_pretrained('/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/sft_ppo/2023-09-26_10-36-41')

# Build the perplexity evaluation pipeline
perplexity = load("perplexity", module_type="measurement")

# DO I NEED TO TAKE CARE OF PAD TOKEN FOR PERPLEXITY PIPELINE AS WELL??
"""
# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id
"""
device = sft_tester.accelerator.device
if sft_tester.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = sft_tester.accelerator.state.deepspeed_plugin
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
generation_kwargs = {
    "min_length": 48,
    "top_k": 50,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 48,
}


#date_n_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#os.makedirs(f'/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/evaluation/{date_n_time}')

model_names_all = ['zero_shot_model', 'sft_model', 'ppo_model', 'sft_ppo_model']
testers_all = [zero_shot_tester, sft_tester, sft_ppo_tester] #ppo_tester,

# infer on test dataset
#test_scores_1, test_scores_2, test_scores_3, test_scores_4 = [], [], [], []
positive_score_all = []
negative_score_all = []
label_all = []
critical_pos_texts = []
critical_neg_texts = []
critical_neg_scores = []
critical_pos_scores = []
total_critical_texts = []
total_critical_scores = []
total_critical_labels = []

for epoch, batch in tqdm(enumerate(zero_shot_tester.dataloader)):
    query_tensors = batch["input_ids"]
    
    texts = batch["review"]
    texts = [text[:1000 ] for text in texts]
    #texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]

    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    pos_score = [output[1]["score"] for output in pipe_outputs]
    positive_score_all.append(pos_score)

    neg_score = [output[0]["score"] for output in pipe_outputs]
    negative_score_all.append(neg_score)

    labels = [label.item() for label in batch["label"]]
    label_all.append(labels)

    for ii in range(len(label_all[0])):
        if labels[ii] == 0 and pos_score[ii] > 0:
            critical_neg_texts.append(texts[ii])
            critical_neg_scores.append(pos_score[ii])
            total_critical_texts.append(texts[ii])
            total_critical_scores.append(pos_score[ii])
            total_critical_labels.append(labels[ii])
        if labels[ii] == 1 and pos_score[ii] < 0:
            critical_pos_texts.append(texts[ii])
            critical_pos_scores.append(pos_score[ii])
            total_critical_texts.append(texts[ii])
            total_critical_scores.append(pos_score[ii])
            total_critical_labels.append(labels[ii])

print(f'Total critical reviews: {len(total_critical_labels)}')
# get flattened lists 
pos_scores = list(itertools.chain(*positive_score_all))
neg_scores = list(itertools.chain(*negative_score_all))
labels = list(itertools.chain(*label_all))

pdb.set_trace()
# plot positive and negative score and save the plot
df = pd.DataFrame(list(zip(neg_scores, pos_scores, labels)), columns = ['neg score', 'pos score', 'label'])
ax = df.plot.scatter(x='neg score', y='pos score', c='label', colormap='viridis')
ax.plot([3,-3], [-3,3], 'ro-')
#fig = plt.scatter(x=df['neg score'], y=df['pos score'], c=df['label'], colormap='viridis')
ax.grid(True)
fig = ax.get_figure()
fig.savefig('scatter_plot.png')

# Plot histograms
fig2, axs2 = plt.subplots(1, 2, figsize=(10,5))

df[df['label'] == 0]['neg score'].plot.hist(ax=axs2[0], bins=10, alpha=0.5, label='Class 0')
df[df['label'] == 1]['neg score'].plot.hist(ax=axs2[0], bins=10, alpha=0.5, label='Class 1')
axs2[0].set_title('Negative Score Distribution')
axs2[0].legend()

df[df['label'] == 0]['pos score'].plot.hist(ax=axs2[1], bins=10, alpha=0.5, label='Class 0') 
df[df['label'] == 1]['pos score'].plot.hist(ax=axs2[1], bins=10, alpha=0.5, label='Class 1')
axs2[1].set_title('Positive Score Distribution')
axs2[1].legend()

fig2.savefig('histograms_plot.png')

# save critical reviews as csv 
df1 = pd.DataFrame({'score': critical_neg_scores, 'review': critical_neg_texts})
df2 = pd.DataFrame({'score': critical_pos_scores, 'review': critical_pos_texts})
df3 = pd.DataFrame({'review': total_critical_texts})

df1.to_csv('critical_neg_texts.csv')
df2.to_csv('critical_pos_texts.csv')
df3.to_csv('total_critical_texts.csv')

# histogram plot for scores of critical reviews 
df4 = pd.DataFrame(list(zip(total_critical_scores, total_critical_labels)), columns = ['pos score', 'label'])
fig3, axs3 = plt.subplots()

df4[df4['label'] == 0]['pos score'].plot.hist(ax=axs3, bins=5, alpha=0.5, label='Class 0') 
df4[df4['label'] == 1]['pos score'].plot.hist(ax=axs3, bins=5, alpha=0.5, label='Class 1')
axs3.set_title('Positive Score Distribution')
axs3.legend()

fig3.savefig('critical_reviews_scores_1000(64+186_tokens).png')