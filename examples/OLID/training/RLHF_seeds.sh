# python ppo_auth2.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="RLHF <20,32> seed 36" \
#          --prompt_len=20 \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_max_new_tokens=32 \
#          --ppo_config.seed=36

# python ppo_auth2.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="RLHF <20,32> seed 12" \
#          --prompt_len=20 \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_max_new_tokens=32 \
#          --ppo_config.seed=12

python ppo_auth2.py \
         --input_query_size="more" \
         --ppo_config.exp_name="RLHF <35,32> seed 36" \
         --prompt_len=35 \
         --generation_kwargs_min_length=32 \
         --generation_kwargs_max_new_tokens=32 \
         --ppo_config.seed=36

python ppo_auth2.py \
         --input_query_size="more" \
         --ppo_config.exp_name="RLHF <35,32> seed 12" \
         --prompt_len=35 \
         --generation_kwargs_min_length=32 \
         --generation_kwargs_max_new_tokens=32 \
         --ppo_config.seed=12