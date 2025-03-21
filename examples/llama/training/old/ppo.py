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

import pdb
import pickle
import datetime
import os
from transformers import GPT2Tokenizer, GPT2Model

date_n_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(date_n_time)
logging_dir = f"/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/ppo/{date_n_time}"
os.makedirs(logging_dir)

@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="gpt2",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1.41e-5, #1e-6, #
            mini_batch_size=128,
            batch_size=128,
            gradient_accumulation_steps=1, #1280, #The number of gradient accumulation steps
            early_stopping=False,
            target_kl=6.0, #1.0,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
            #steps= 128000, #64000,
            #init_kl_coef=0.1,
            #vf_coef=0.5,
            #ppo_epochs=5,
            #gamma=0.99, #discount factor = gamma? 
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

    def input_size():
        return 64 #LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(args.ppo_config, args.query_dataset)
test_dataset = build_dataset(args.ppo_config, args.query_dataset, data_split='test')

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
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.pad_token_as_eos_token = True
tokenizer.max_length = 64

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
# Creating a tester object here because dataloader is only available in the PPOTrainer class
ppo_tester = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=test_dataset, data_collator=collator)

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
generation_kwargs = {
    "min_length": 48,
    "top_k": 50,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 48,
}

# stat lists
obj_entropy = []
policy_entropy = []

mean_non_score_reward = []
mean_scores = []
std_scores = []

mean_returns = []
var_returns = []

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]

    # including the line below to deal with sentiment_pipeline's tokenizer
    texts = [text[:512] if len(text) > 512 else text for text in texts]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    
    """
    rewards = [
        torch.nn.functional.softmax(torch.tensor([output[0]["score"], output[1]["score"]]))[1]
        for output in pipe_outputs
    ]
    """
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    
    # print stats
    print(f'input:{batch["query"][:3]}, output:{batch["response"][:3]} \n\n')
    
    obj_entropy.append(stats['objective/entropy'])
    policy_entropy.append(stats['ppo/policy/entropy'])
    mean_non_score_reward.append(stats['ppo/mean_non_score_reward'])
    mean_scores.append(stats['ppo/mean_scores'])
    std_scores.append(stats['ppo/std_scores'])
    mean_returns.append(stats['ppo/returns/mean'])
    var_returns.append(stats['ppo/returns/var'])

    print(f'epoch: {epoch}')
    mean_return = stats['ppo/returns/mean']
    mean_score = stats['ppo/mean_scores']
    print(f'mean_returns: {mean_return}, mean_score: {mean_score}')


# save the final model - maybe this will unnecessarily take memory? What is a good way to go about it? 

#torch.save(model, logging_dir +'/model.pt')
#ppo_trainer.create_model_card(f'/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/ppo/{date_n_time}')
#ppo_trainer._save_pretrained(logging_dir)

# save the train stats 
#train_stats_lists = [obj_entropy, policy_entropy, mean_non_score_reward, mean_scores, std_scores, mean_returns, var_returns]
#train_stats_lists_names = ['obj_entropy', 'policy_entropy', 'mean_non_score_reward', 'mean_scores', 'std_scores', 'mean_returns', 'var_returns']
#for ii in range(len(train_stats_lists)):
#    pickle.dump(train_stats_lists[ii], open(logging_dir + f'/{train_stats_lists_names[ii]}', "wb"))


"""
# infer on test dataset
test_scores = []
for epoch, batch in tqdm(enumerate(ppo_tester.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [
        torch.nn.functional.softmax(torch.tensor([output[0]["score"], output[1]["score"]]))[1]
        for output in pipe_outputs
    ]

    test_scores.append(sum(rewards)/len(rewards))
    print(sum(rewards)/len(rewards))

# print and save the final score  
final_test_score = sum(test_scores)/len(test_scores)     
print(f'Total test sentiment-score is: {final_test_score.item()}')
pickle.dump(final_test_score.item(), open(f'/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/ppo/{date_n_time}/final_test_score', "wb"))
"""

    




