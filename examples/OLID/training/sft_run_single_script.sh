#!/bin/bash
# chmod +x run_scripts.sh
# ./run_scripts.sh 

# 64,48 with trl parameters
# python sft_auth1.py \
#          --input_query_size="more" \
#          --exp_name="64,48 with trl parameters: seed=0" \
#          --generation_kwargs_min_length=48 \
#          --generation_kwargs_max_new_tokens=48 


# 64,48 with rl4lms parameters: seed=0
python sft_auth2.py \
         --input_query_size="more" \
         --exp_name="OLID SFT : seed=2" \
         --generation_kwargs_min_length=32 \
         --generation_kwargs_top_k=0 \
         --generation_kwargs_top_p=1.0 \
         --generation_kwargs_max_new_tokens=32 \
         --seed=2

# python sft_auth2.py \
#          --input_query_size="more" \
#          --exp_name="20,32 with rl4lms parameters: seed=36" \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_top_k=0 \
#          --generation_kwargs_top_p=1.0 \
#          --generation_kwargs_max_new_tokens=32 \
#          --seed=36

# python sft_auth2.py \
#          --input_query_size="more" \
#          --exp_name="20,32 with rl4lms parameters: seed=12" \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_top_k=0 \
#          --generation_kwargs_top_p=1.0 \
#          --generation_kwargs_max_new_tokens=32 \
#          --seed=12

# full python command
# python sft_auth1.py \
#          --input_query_size="more" \
#          --exp_name="64,48 with rl4lms parameters: seed=0" \
#          --generation_kwargs_min_length=48 \
#          --generation_kwargs_top_k=50 \
#          --generation_kwargs_top_p=1.0 \
#          --generation_kwargs_max_new_tokens=48 \
#          --change_tokenizer_args 