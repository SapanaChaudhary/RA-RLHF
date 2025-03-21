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
from datasets import load_dataset, Dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)
from peft import LoraConfig
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    create_reference_model,
    set_seed,
)
from trl.trainer.ppo_trainer import PPOTrainer
from trl.core import LengthSampler
import datetime
import os
import pandas as pd

date_n_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging_dir = f"/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/big/imdb/{date_n_time}"
os.makedirs(logging_dir, exist_ok=True)

tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPTJ model to generate less toxic contents
# by using allenai/real-toxicity-prompts dataset. We use PPO
#  (proximal policy optimization) to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="ybelkada/gpt-j-6b-sharded-bf16", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=8, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    seed: Optional[int] = field(default=42, metadata={"help": "the seed for the experiment"})
    exp_name: Optional[str] = field(default="imdb", metadata={"help": "the name of the experiment"})
    query_dataset: Optional[str] = field(default="imdb", metadata={"help": "the name of the dataset"})
    risk_n: int = field(
        default=300, metadata={"help": "240 (> batch size): no RA; 70: RA begins after iter 70; 1: RA throughout"}
    )
    risk_alpha: Optional[float] = field(default=0.2, metadata={"help": "risk alpha value = 20 percent"})
    risk_rho: Optional[float] = field(
        default=0.95, metadata={"help": "risk alpha reaches a value = 20 percent at 80 percent of total iterations"}
    )


model_save_path = os.path.join(logging_dir, "save_pretrained")
prompt_len = 64
gen_len = 48

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    project_kwargs={"logging_dir": logging_dir},
    # ppo_epochs=100,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    seed=script_args.seed,
    exp_name=script_args.exp_name,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    bias="none",
    task_type="CAUSAL_LM",
)


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

    def input_size():
        return prompt_len

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config, script_args.query_dataset)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer. We first load the model
# in bfloat16 to save memory using `transformers`.
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
# And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
model = AutoModelForCausalLMWithValueHead.from_pretrained(model, peft_config=peft_config)

# We create a reference model by sharing 20 layers
# ref_model = create_reference_model(model, num_shared_layers=20)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/sft_auth2/IMDB/2024-02-18_22-24-38",
    trust_remote_code=True,
)  # sft positive 70_30 gpt2

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the reward pipeline, we will use the toxicity model to compute the reward.
# We first load the toxicity model and tokenizer.
# toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
# toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# # We load the toxicity model in fp16 to save memory.
# toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).to(
#     ppo_trainer.accelerator.device
# )

toxicity_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
toxicity_model = AutoModelForSequenceClassification.from_pretrained(
    "lvwerra/distilbert-imdb", torch_dtype=torch.float16
).to(ppo_trainer.accelerator.device)


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": gen_len,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
# output_min_length = 20
# output_max_length = 30
# output_length_sampler = LengthSampler(output_min_length, output_max_length)

# model_save_path = script_args.model_save_path
import math


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
    alpha = script_args.risk_alpha
    rho = script_args.risk_rho

    # if script_args.risk_scheduler == "old":
    #     if m <= script_args.risk_n:
    #         return script_args.ppo_config.batch_size
    #     else:
    #         return math.ceil(script_args.ppo_config.batch_size * max(alpha, 1 - (1 - alpha) * (m - script_args.risk_n) / (rho * M)))
    # else:
    # print("here")
    if m <= script_args.risk_n:
        val = script_args.batch_size
    elif m >= math.ceil(rho * M):
        val = math.ceil(alpha * script_args.batch_size)
    else:
        K = (1 - alpha) / (math.ceil(rho * M) - script_args.risk_n)
        val = math.ceil(script_args.batch_size * max(alpha, 1 - K * (m - script_args.risk_n)))
    return val


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        # gen_len = output_length_sampler()
        # generation_kwargs["max_new_tokens"] = gen_len
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score # noqa
    texts = batch["response"]
    toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        ppo_trainer.accelerator.device
    )
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 1]).tolist()

    rewards = [torch.tensor(output) for output in toxicity_labels]

    alpha_N_ceil = get_current_risk_level(PPOConfig.soft_risk_alpha, PPOConfig.risk_level_scheduler, epoch, 3601)

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards, alpha_N_ceil)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)
ppo_trainer.save_pretrained(model_save_path)
