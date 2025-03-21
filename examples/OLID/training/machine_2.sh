#seed=2
# python soft_risk_ppo_auth2_seed2.py \
#          --input_query_size="more" \
#          --ppo_config.exp_name="RA-RLHF; seed 2  70/30" \
#          --prompt_len=8 \
#          --generation_kwargs_min_length=32 \
#          --generation_kwargs_max_new_tokens=32 \
#          --ppo_config.seed=2 \
#          --risk_scheduler="new" \
#          --risk_n=70 \
#          --risk_alpha=0.4 \
#          --risk_rho=0.95

# python soft_risk_ppo_auth2_seed2.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name="RA-RLHF; seed 12 70/30" \
#         --prompt_len=8 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=12 \
#         --risk_scheduler="new" \
#         --risk_n=70 \
#         --risk_alpha=0.4 \
#         --risk_rho=0.95
python soft_risk_ppo_auth2_seed2.py \
        --input_query_size="more" \
        --ppo_config.exp_name="alhpa 0.2 RA-RLHF; seed 4 OLID correct" \
        --prompt_len=8 \
        --generation_kwargs_min_length=22 \
        --generation_kwargs_max_new_tokens=22 \
        --ppo_config.seed=4 \
        --risk_scheduler="new" \
        --risk_n=30 \
        --risk_alpha=0.2 \
        --risk_rho=0.95
python soft_risk_ppo_auth2_seed2.py \
        --input_query_size="more" \
        --ppo_config.exp_name="alhpa 0.2 RA-RLHF; seed 56 OLID correct" \
        --prompt_len=8 \
        --generation_kwargs_min_length=22 \
        --generation_kwargs_max_new_tokens=22 \
        --ppo_config.seed=56 \
        --risk_scheduler="new" \
        --risk_n=30 \
        --risk_alpha=0.2 \
        --risk_rho=0.95
python soft_risk_ppo_auth2_seed2.py \
        --input_query_size="more" \
        --ppo_config.exp_name="alhpa 0.2 RA-RLHF; seed 92 OLID correct" \
        --prompt_len=8 \
        --generation_kwargs_min_length=22 \
        --generation_kwargs_max_new_tokens=22 \
        --ppo_config.seed=92 \
        --risk_scheduler="new" \
        --risk_n=30 \
        --risk_alpha=0.2 \
        --risk_rho=0.95

# python soft_risk_ppo_auth2_seed2.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name="alhpa 0.2 RA-RLHF; seed 56 70/30" \
#         --prompt_len=8 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=56 \
#         --risk_scheduler="new" \
#         --risk_n=30 \
#         --risk_alpha=0.2 \
#         --risk_rho=0.95

# python soft_risk_ppo_auth2_seed2.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name="RA-RLHF; seed 56 70/30" \
#         --prompt_len=8 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=56 \
#         --risk_scheduler="new" \
#         --risk_n=70 \
#         --risk_alpha=0.4 \
#         --risk_rho=0.95

# python soft_risk_ppo_auth2_seed12.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name=" RA-RLHF <8,32> seed 12 truly" \
#         --prompt_len=8 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=12 \
#         --risk_scheduler="new" \
#         --risk_n=30 \
#         --risk_alpha=0.4 \
#         --risk_rho=0.95

# python soft_risk_ppo_auth2_seed36.py \
#         --input_query_size="more" \
#         --ppo_config.exp_name=" RA-RLHF <8,32> seed 36 truly" \
#         --prompt_len=8 \
#         --generation_kwargs_min_length=32 \
#         --generation_kwargs_max_new_tokens=32 \
#         --ppo_config.seed=36 \
#         --risk_scheduler="new" \
#         --risk_n=30 \
#         --risk_alpha=0.4 \
#         --risk_rho=0.95


