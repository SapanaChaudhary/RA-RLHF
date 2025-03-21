python soft_risk_ppo_auth2_2.py \
         --input_query_size="more" \
         --ppo_config.exp_name=" RA-RLHF <15,32> alpha 30, n 30, rho 95 " \
         --prompt_len=15 \
         --generation_kwargs_min_length=32 \
         --generation_kwargs_max_new_tokens=32 \
         --ppo_config.seed=42 \
         --risk_scheduler="new" \
         --risk_n=30 \
         --risk_alpha=0.3 \
         --risk_rho=0.95

python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <15,45> alpha 30, n 30, rho 95 " \
            --prompt_len=15 \
            --generation_kwargs_min_length=45 \
            --generation_kwargs_max_new_tokens=45 \
            --ppo_config.seed=42 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.3 \
            --risk_rho=0.95

python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <20,32> alpha 30, n 30, rho 95 " \
            --prompt_len=20 \
            --generation_kwargs_min_length=32 \
            --generation_kwargs_max_new_tokens=32 \
            --ppo_config.seed=42 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.3 \
            --risk_rho=0.95

python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <20,45> alpha 30, n 30, rho 95 " \
            --prompt_len=20 \
            --generation_kwargs_min_length=45 \
            --generation_kwargs_max_new_tokens=45 \
            --ppo_config.seed=42 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.3 \
            --risk_rho=0.95

python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <35,32> alpha 30, n 30, rho 95 " \
            --prompt_len=35 \
            --generation_kwargs_min_length=32 \
            --generation_kwargs_max_new_tokens=32 \
            --ppo_config.seed=42 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.3 \
            --risk_rho=0.95

python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <35,45> alpha 30, n 30, rho 95 " \
            --prompt_len=35 \
            --generation_kwargs_min_length=45 \
            --generation_kwargs_max_new_tokens=45 \
            --ppo_config.seed=42 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.3 \
            --risk_rho=0.95