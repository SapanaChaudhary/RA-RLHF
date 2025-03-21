#!/bin/bash
# chmod +x run_scripts.sh
# ./run_scripts.sh 

# 64,48 with trl parameters
python single_model_generic.py \
         --ppo_config.model_name="/mnt/shared-scratch/Anon1/auth1/rlhf/trl/logs/trl_sft_ppo/2023-10-23_23-30-47"
         --input_query_size="more" \
         --exp_name="64,48 testing"  \
         --reward_function="rl4lms" \
         --generation_kwargs_min_length=48 \
         --generation_kwargs_top_k=50 \
         --generation_kwargs_top_p=1.0 \
         --generation_kwargs_max_new_tokens=48 \
         --change_tokenizer_args

# # 64,48 with trl parameters
# python single_model_generic.py \
#          --input_query_size="more" \
#          --exp_name="64,48 with rl4lms parameters: seed=42" \
#          --generation_kwargs_min_length=48 \
#          --generation_kwargs_top_k=50 \
#          --generation_kwargs_top_p=1.0 \
#          --generation_kwargs_max_new_tokens=48 \
#          --change_tokenizer_args


# full python command
# python single_model_generic.py \
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