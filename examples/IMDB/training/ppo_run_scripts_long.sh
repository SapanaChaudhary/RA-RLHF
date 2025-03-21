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

# seed = 0
# 64,20 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,20 with trl parameters" \
         --generation_kwargs_min_length=20 \
         --generation_kwargs_max_new_tokens=20 

# 64,30 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,30 with trl parameters" \
         --generation_kwargs_min_length=30 \
         --generation_kwargs_max_new_tokens=30 

# 64,48 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,48 with trl parameters" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_max_new_tokens=48 

# 64,75 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,75 with trl parameters" \
         --generation_kwargs_min_length=75 \
         --generation_kwargs_max_new_tokens=75

# 64,100 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,100 with trl parameters" \
         --generation_kwargs_min_length=100 \
         --generation_kwargs_max_new_tokens=100 

# 64,130 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,130 with trl parameters" \
         --generation_kwargs_min_length=130 \
         --generation_kwargs_max_new_tokens=130

# 64,160 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,160 with trl parameters" \
         --generation_kwargs_min_length=160 \
         --generation_kwargs_max_new_tokens=160

# 64,200 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,200 with trl parameters" \
         --generation_kwargs_min_length=200 \
         --generation_kwargs_max_new_tokens=200 

# 64,300 with trl parameters
python ppo_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,300 with trl parameters" \
         --generation_kwargs_min_length=300 \
         --generation_kwargs_max_new_tokens=300 

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