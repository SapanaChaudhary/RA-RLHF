#!/bin/bash
# chmod +x run_scripts.sh
# ./run_scripts.sh

# (2-8),32 with trl reward and rl4lm tokenizer parameters
python sentiment_tuning_auth1.py \
         --ppo_config.exp_name="(2-8),32 with trl reward and rl4lm tokenizer parameters" \
         --change_tokenizer_args 

# (2-8),32 with trl reward; rl4lm generation and tokenizer parameters
python sentiment_tuning_auth1.py \
         --ppo_config.exp_name="(2-8),32 with trl reward; rl4lm generation and tokenizer parameters" \
         --generation_kwargs.min_length=32 \
         --generation_kwargs.top_k=50 \
         --generation_kwargs.top_p=1.0 \
         --generation_kwargs.max_new_tokens=32 \
         --change_tokenizer_args 


# 64,32 with trl parameters
python sentiment_tuning_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,32 with trl parameters" 


# 64,32 with topk=50
python sentiment_tuning_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,32 with topk=50" \
         --generation_kwargs.min_length=32 \
         --generation_kwargs.top_k=50 \
         --generation_kwargs.top_p=1.0 \
         --generation_kwargs.max_new_tokens=32 

# 64,32 with rl4lms tokenizer args
python sentiment_tuning_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,32 with rl4lms tokenizer args" \
         --generation_kwargs.min_length=32 \
         --generation_kwargs.top_k=0.0 \
         --generation_kwargs.top_p=1.0 \
         --generation_kwargs.max_new_tokens=32 \
         --change_tokenizer_args 

# 64,32 with rl4lms args
python sentiment_tuning_auth1.py \
         --input_query_size="more" \
         --ppo_config.exp_name="64,32 with rl4lms args" \
         --generation_kwargs.min_length=32 \
         --generation_kwargs.top_k=50 \
         --generation_kwargs.top_p=1.0 \
         --generation_kwargs.max_new_tokens=32 \
         --change_tokenizer_args 

# full python command
# python examples/scripts/sentiment_tuning_auth1.py \
#          --model_name="/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/sft_positive_1/oct_2_afternoon/checkpoint-120" \
#          --input_query_size="more" \
#          --ppo_config.exp_name="(2-8),32 rl4lm tokenizer" \
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