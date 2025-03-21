python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <20,32> alpha 20, n 30, rho 95 ; seed 36" \
            --prompt_len=20 \
            --generation_kwargs_min_length=32 \
            --generation_kwargs_max_new_tokens=32 \
            --ppo_config.seed=36 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.2 \
            --risk_rho=0.95

python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <20,32> alpha 20, n 30, rho 95 ; seed 12" \
            --prompt_len=20 \
            --generation_kwargs_min_length=32 \
            --generation_kwargs_max_new_tokens=32 \
            --ppo_config.seed=12 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.2 \
            --risk_rho=0.95



python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <35,32> alpha 40, n 30, rho 95; seed 36 " \
            --prompt_len=35 \
            --generation_kwargs_min_length=32 \
            --generation_kwargs_max_new_tokens=32 \
            --ppo_config.seed=36 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.4 \
            --risk_rho=0.95

python soft_risk_ppo_auth2_2.py \
            --input_query_size="more" \
            --ppo_config.exp_name=" RA-RLHF <35,32> alpha 40, n 30, rho 95; seed 12 " \
            --prompt_len=35 \
            --generation_kwargs_min_length=32 \
            --generation_kwargs_max_new_tokens=32 \
            --ppo_config.seed=12 \
            --risk_scheduler="new" \
            --risk_n=30 \
            --risk_alpha=0.4 \
            --risk_rho=0.95