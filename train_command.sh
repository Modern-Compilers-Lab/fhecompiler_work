python -m fhe_rl train \
  --dataset ./fhe_rl/datasets/my_data.txt \
  --n_envs 8 \
  --total_timesteps 2000000 \
  --n_cycle 1 \
  --n_budget 5 \
  --lambda_env 0.1 \
  --lambda_kl 0.05