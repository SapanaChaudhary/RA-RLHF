python soft_risk_ppo_auth2_2.py \
         --input_query_size="more" \
         --ppo_config.exp_name="RA-RLHF_alpha_5" \
         --generation_kwargs_min_length=32 \
         --generation_kwargs_max_new_tokens=32 \
         --ppo_config.seed=42 \
         --risk_scheduler="new" \
         --risk_n=1 \
         --risk_alpha=0.05 \
         --risk_rho=0.80

python soft_risk_ppo_auth2_2.py \
         --input_query_size="more" \
         --ppo_config.exp_name="RA-RLHF_alpha_20" \
         --generation_kwargs_min_length=32 \
         --generation_kwargs_max_new_tokens=32 \
         --ppo_config.seed=42 \
         --risk_scheduler="new" \
         --risk_n=1 \
         --risk_alpha=0.2 \
         --risk_rho=0.80

python soft_risk_ppo_auth2_2.py \
         --input_query_size="more" \
         --ppo_config.exp_name="RA-RLHF_alpha_45" \
         --generation_kwargs_min_length=32 \
         --generation_kwargs_max_new_tokens=32 \
         --ppo_config.seed=42 \
         --risk_scheduler="new" \
         --risk_n=1 \
         --risk_alpha=0.45 \
         --risk_rho=0.80