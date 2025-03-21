# Does bad on seed 42
# python ppo_auth2.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="RLHF <8,32> seed 42" \
#          --prompt_len=8 \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_max_new_tokens=32 \
#          --ppo_config.seed=42

# python ppo_auth2.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="RLHF <8,32> seed 36" \
#          --prompt_len=8 \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_max_new_tokens=32 \
#          --ppo_config.seed=36

# python ppo_auth2.py \
#             --input_query_size="more" \
#             --ppo_config.exp_name="RLHF <8,32> seed 12" \
#             --prompt_len=8 \
#             --generation_kwargs_min_length=32 \
#             --generation_kwargs_max_new_tokens=32 \
#             --ppo_config.seed=12

# python ppo_auth2_seed2.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="RLHF; seed 2 70/30" \
#          --prompt_len=8 \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_max_new_tokens=32 \
#          --ppo_config.seed=2

python ppo_auth2_seed2.py \
        --input_query_size="more" \
        --ppo_config.exp_name="RLHF; seed 4 Real_toxicity 2x" \
        --prompt_len=32 \
        --generation_kwargs_min_length=32 \
        --generation_kwargs_max_new_tokens=32 \
        --ppo_config.seed=4

# python ppo_auth2_seed2.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name="RLHF; seed 56 Real_toxicity 2x" \
#         --prompt_len=32 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=56

# python ppo_auth2_seed2.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name="RLHF; seed 92 Real_toxicity 2x" \
#         --prompt_len=32 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=92

# python ppo_auth2_seed12.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name="Longer RLHF <8,32> seed 12 truly" \
#         --prompt_len=8 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=12

# python ppo_auth2_seed36.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name="RLHF <8,32> seed 36 truly" \
#         --prompt_len=8 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=36