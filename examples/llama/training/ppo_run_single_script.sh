#!/bin/bash
# chmod +x run_scripts.sh
# ./run_scripts.sh 

# # 64,32 with trl parameters
# python ppo_auth1.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="64,32 with trl parameters"

# # 64,32 with trl parameters + rl4lm gen
# python ppo_auth1.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="64,32 with with trl parameters + rl4lm gen" \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_top_k=50 \
#          --generation_kwargs_top_p=1.0 \
#          --generation_kwargs_max_new_tokens=32

# # 64,32 with trl parameters + rl4lm tokenizer
# python ppo_auth1.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="64,32 with with trl parameters + rl4lm tokenizer" \
#          --change_tokenizer_args

# # 64,32 with trl parameters + + rl4lm gen + rl4lm tokenizer
# python ppo_auth1.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="64,32 with trl parameters + rl4lm gen + rl4lm tokenizer" \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_top_k=50 \
#          --generation_kwargs_top_p=1.0 \
#          --generation_kwargs_max_new_tokens=32 \
#          --change_tokenizer_args

# seed = 42
# 64,48 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with trl parameters; seed=42" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_max_new_tokens=48 \
         --ppo_config.seed=42

# 64,48 with trl parameters + rl4lm gen
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with with trl parameters + rl4lm gen; seed=42" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_top_k=50 \
         --generation_kwargs_top_p=1.0 \
         --generation_kwargs_max_new_tokens=48 \
         --ppo_config.seed=42

# 64,48 with trl parameters + rl4lm tokenizer
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with with trl parameters + rl4lm tokenizer; seed=42" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_max_new_tokens=48 \
         --change_tokenizer_args \
         --ppo_config.seed=42

# 64,48 with trl parameters + rl4lm gen + rl4lm tokenizer
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with trl parameters + rl4lm gen + rl4lm tokenizer; seed=42" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_top_k=50 \
         --generation_kwargs_top_p=1.0 \
         --generation_kwargs_max_new_tokens=48 \
         --change_tokenizer_args \
         --ppo_config.seed=42

# seed = 73
# 64,48 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with trl parameters; seed=73" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_max_new_tokens=48 \
         --ppo_config.seed=73

# 64,48 with trl parameters + rl4lm gen
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with with trl parameters + rl4lm gen; seed=73" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_top_k=50 \
         --generation_kwargs_top_p=1.0 \
         --generation_kwargs_max_new_tokens=48 \
         --ppo_config.seed=73

# 64,48 with trl parameters + rl4lm tokenizer
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with with trl parameters + rl4lm tokenizer; seed=73" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_max_new_tokens=48 \
         --change_tokenizer_args \
         --ppo_config.seed=73

# 64,48 with trl parameters + rl4lm gen + rl4lm tokenizer
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with trl parameters + rl4lm gen + rl4lm tokenizer; seed=73" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_top_k=50 \
         --generation_kwargs_top_p=1.0 \
         --generation_kwargs_max_new_tokens=48 \
         --change_tokenizer_args \
         --ppo_config.seed=73

# full python command
# python ppo_auth1.py \
#          --model_name="/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/sft_positive_1/oct_2_afternoon/checkpoint-120" \
#          --input_query_size="more" \
#          --exp_name="(2-8),32 rl4lm tokenizer" \
#          --generation_kwargs.min_length \
#          --generation_kwargs.top_k \
#          --generation_kwargs.top_p \
#          --generation_kwargs.do_sample \
#          --generation_kwargs.max_new_tokens \
#          --change_tokenizer_args=True \
#          --tokenizer_kwargs.padding_side \
#          --tokenizer_kwargs.truncation_side \
#          --tokenizer_kwargs.pad_token_as_eos_token \
#          --tokenizer_kwargs.max_length \
#          --reward_function="rl4lm"