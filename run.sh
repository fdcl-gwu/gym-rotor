# sh run.sh 
# --seed
# --aux_id
# --wrapper_id
# --discount
# --lr
# --actor_hidden_dim
# --critic_hidden_dim
# --batch_size

# lam_T = 0.5 # 0.5 - 0.8
# lam_S = 0.5
# lam_M = 0.5 # 0.2 - 0.5

python3 main.py --seed 1     --lr 3e-4 --discount 0.99 --actor_hidden_dim 32 --critic_hidden_dim 256 --batch_size 256 
python3 main.py --seed 12    --lr 3e-4 --discount 0.99 --actor_hidden_dim 32 --critic_hidden_dim 256 --batch_size 256
python3 main.py --seed 123   --lr 3e-4 --discount 0.99 --actor_hidden_dim 32 --critic_hidden_dim 256 --batch_size 256
python3 main.py --seed 1234  --lr 3e-4 --discount 0.99 --actor_hidden_dim 32 --critic_hidden_dim 256 --batch_size 256
python3 main.py --seed 12345 --lr 3e-4 --discount 0.99 --actor_hidden_dim 32 --critic_hidden_dim 256 --batch_size 256
