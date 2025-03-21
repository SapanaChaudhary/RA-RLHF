from transformers import pipeline, AutoTokenizer
from datasets import Dataset, load_from_disk
from dataclasses import dataclass, field
from typing import Optional
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import pdb
import tyro
import torch
from evaluate import load

# load the dataset
# jigsaw_df_train = pd.read_csv("jigsaw_train.csv")
# jigsaw_df_test = pd.read_csv("jigsaw_test.csv")

# load reward model
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
jigsaw_reward_model = pipeline("sentiment-analysis", model="unitary/toxic-bert", **sent_kwargs)

ref_model_path = "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/sft_auth2/real_toxicity/2024-02-22_14-11-00"  # sft positive
# ref_model_path = (
# "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/sft_auth2/2024-01-20_20-32-35"  # sft both
# )


model_path = "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/real_toxicity/2024-02-22_14-39-57/save_pretrained"
risk_model_path = "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/real_toxicity/2024-02-22_17-34-54/save_pretrained"

# 2x stuff
# model_path = "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/real_toxicity/2024-02-23_11-16-51/save_pretrained"
# risk_model_path = "/mnt/research/Anon2/Students/auth2/rlhf/trl/logs/dummy/real_toxicity/2024-02-23_12-37-19/save_pretrained"

BETA = 0.2
BETA_risk = 0.2
GAMMA = 1.0

prompt_len = 32
response_len = 32
alpha_level = 20
seed = 56


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            # model_name="gpt2",
            query_dataset=None,
            # reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1.41e-5,
            log_with=None,
            mini_batch_size=128,
            batch_size=128,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            init_kl_coef=BETA,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    ppo_config_risk: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            # model_name="gpt2",
            query_dataset=None,
            # reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1.41e-5,
            log_with=None,
            mini_batch_size=128,
            batch_size=128,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            init_kl_coef=BETA_risk,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    query_dataset: str = field(default=None, metadata={"help": "the dataset to query"})
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq models"})
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})


args = tyro.cli(ScriptArguments)

tokenizer = AutoTokenizer.from_pretrained(ref_model_path)
tokenizer.pad_token = tokenizer.eos_token

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ref_model_path,
    trust_remote_code=True,
    device_map=None,
    peft_config=None,
)


model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=None,
    peft_config=None,
)

risk_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    risk_model_path,
    trust_remote_code=True,
    device_map=None,
    peft_config=None,
)
gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "gpt2",
    trust_remote_code=True,
    device_map=None,
    peft_config=None,
)
ppo_trainer_gpt2 = PPOTrainer(args.ppo_config, model=gpt2_model, ref_model=ref_model, tokenizer=tokenizer)
ppo_trainer_sft = PPOTrainer(args.ppo_config, model=ref_model, ref_model=ref_model, tokenizer=tokenizer)
ppo_trainer = PPOTrainer(args.ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer)
ppo_trainer_risk = PPOTrainer(args.ppo_config_risk, model=risk_model, ref_model=ref_model, tokenizer=tokenizer)
perplexity = load("perplexity", module_type="measurement")


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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_from_disk("/mnt/shared-scratch/Shakkottai_S/auth2d36/datasets/RealToxicity/real_toxicity_test.hf")
    ds = ds.rename_columns({"Prompt": "review"})

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


tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

generation_kwargs = {
    "min_length": response_len,
    "top_k": 0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": response_len,
}


batch_size = 128
ds = build_dataset()
df = pd.DataFrame()
for i in tqdm(range(0, len(ds), batch_size)):
    batch = ds[i : i + batch_size]
    query_tensors = batch["input_ids"]
    query_tensors = [q.to(ppo_trainer.current_device) for q in query_tensors]
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    response_tensors_risk = ppo_trainer_risk.generate(query_tensors, return_prompt=False, **generation_kwargs)
    response_tensors_sft = ppo_trainer_sft.generate(query_tensors, return_prompt=False, **generation_kwargs)
    response_tensors_gpt2 = ppo_trainer_gpt2.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["response_risk"] = tokenizer.batch_decode(response_tensors_risk)
    batch["response_sft"] = tokenizer.batch_decode(response_tensors_sft)
    batch["response_gpt2"] = tokenizer.batch_decode(response_tensors_gpt2)

    # pdb.set_trace()
    text_data = [text[:500] for text in batch["review"]]
    # Compute query sentiment score
    pipe_outputs_prompt = jigsaw_reward_model(
        batch["query"],
        **sent_kwargs,
        # truncation=True,
    )
    prompt_scores = [torch.tensor(-output[0]["score"]) for output in pipe_outputs_prompt]
    batch["prompt_score"] = [s.item() for s in prompt_scores]

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = jigsaw_reward_model(
        texts,
        **sent_kwargs,
        truncation=True,
    )
    texts_risk = [q + r for q, r in zip(batch["query"], batch["response_risk"])]
    pipe_outputs_risk = jigsaw_reward_model(
        texts_risk,
        **sent_kwargs,
        truncation=True,
    )
    texts_sft = [q + r for q, r in zip(batch["query"], batch["response_sft"])]
    pipe_outputs_sft = jigsaw_reward_model(
        texts_sft,
        **sent_kwargs,
        truncation=True,
    )
    texts_gpt2 = [q + r for q, r in zip(batch["query"], batch["response_gpt2"])]
    pipe_outputs_gpt2 = jigsaw_reward_model(
        texts_gpt2,
        **sent_kwargs,
        truncation=True,
    )
    scores = [torch.tensor(-output[0]["score"]) for output in pipe_outputs]
    scores_risk = [torch.tensor(-output[0]["score"]) for output in pipe_outputs_risk]
    scores_sft = [torch.tensor(-output[0]["score"]) for output in pipe_outputs_sft]
    scores_gpt2 = [torch.tensor(-output[0]["score"]) for output in pipe_outputs_gpt2]
    batch["R"] = [s.item() for s in scores]
    batch["R_risk"] = [s.item() for s in scores_risk]
    batch["R_sft"] = [s.item() for s in scores_sft]
    batch["R_gpt2"] = [s.item() for s in scores_gpt2]

    # Compute R_bar
    scores = torch.tensor(scores, device=ppo_trainer.current_device)
    scores_risk = torch.tensor(scores_risk, device=ppo_trainer_risk.current_device)

    model_inputs = ppo_trainer.prepare_model_inputs(query_tensors, response_tensors)
    model_inputs_risk = ppo_trainer_risk.prepare_model_inputs(query_tensors, response_tensors_risk)

    model_inputs_names = list(model_inputs.keys())

    with torch.no_grad():
        all_logprobs, _, values, masks = ppo_trainer.batched_forward_pass(
            ppo_trainer.model,
            query_tensors,
            response_tensors,
            model_inputs,
            response_masks=None,
            return_logits=False,
        )
        all_logprobs_risk, _, values_risk, masks_risk = ppo_trainer_risk.batched_forward_pass(
            ppo_trainer_risk.model,
            query_tensors,
            response_tensors_risk,
            model_inputs_risk,
            response_masks=None,
            return_logits=False,
        )
        ref_logprobs, _, _, _ = ppo_trainer.batched_forward_pass(
            ppo_trainer.ref_model, query_tensors, response_tensors, model_inputs, return_logits=None
        )
        ref_logprobs_risk, _, _, _ = ppo_trainer_risk.batched_forward_pass(
            ppo_trainer_risk.ref_model, query_tensors, response_tensors_risk, model_inputs_risk, return_logits=None
        )
        rewards, non_score_reward = ppo_trainer.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
        rewards_risk, non_score_reward_risk = ppo_trainer_risk.compute_rewards(
            scores_risk, all_logprobs_risk, ref_logprobs_risk, masks_risk
        )
    if GAMMA < 1.0:
        for t in range(rewards.shape[1]):
            rewards[:, t] = GAMMA**t * rewards[:, t]
            rewards_risk[:, t] = GAMMA**t * rewards_risk[:, t]

    batch["R_bar"] = rewards.sum(axis=1).tolist()
    batch["R_bar_risk"] = rewards_risk.sum(axis=1).tolist()

    # Compute perplexity
    batch["perplexity"] = perplexity.compute(data=text_data, model_id=model_path)["perplexities"]
    batch["perplexity_risk"] = perplexity.compute(data=text_data, model_id=risk_model_path)["perplexities"]
    batch["perplexity_sft"] = perplexity.compute(data=text_data, model_id=ref_model_path)["perplexities"]
    batch["perplexity_gpt2"] = perplexity.compute(data=text_data, model_id="gpt2", batch_size=64)["perplexities"]
    # pdb.set_trace()
    try:
        df = pd.concat([df, pd.DataFrame(batch)], ignore_index=True)
    except:
        batch["input_ids"] = [None] * len(batch["review"])
        df = pd.concat([df, pd.DataFrame(batch)], ignore_index=True)


print("Saving results")
df.to_csv(f"realtoxic_generations_{prompt_len}_{response_len}_alpha_{alpha_level}_seed_{seed}_final.csv", index=False)


print("Average RLHF reward:", df["R"].mean())
print("Average RA-RLHF reward:", df["R_risk"].mean())
print("Average SFT reward:", df["R_sft"].mean())
print("Average GPT2 reward:", df["R_gpt2"].mean())

print("Average RLHF perplexity:", df["perplexity"].mean())
print("Average RA-RLHF perplexity:", df["perplexity_risk"].mean())
print("Average SFT perplexity:", df["perplexity_sft"].mean())
print("Average GPT2 perplexity:", df["perplexity_gpt2"].mean())
