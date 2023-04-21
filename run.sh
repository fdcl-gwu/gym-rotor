# sh run.sh 
# --seed
# --aux_id
# --wrapper_id
# --discount
# --lr
# --actor_hidden_dim
# --critic_hidden_dim
# --batch_size

python3 main.py --seed 7     --lr 1e-4 --discount 0.99 --actor_hidden_dim 16 --critic_hidden_dim 512 --batch_size 256 
python3 main.py --seed 12    --lr 1e-4 --discount 0.99 --actor_hidden_dim 16 --critic_hidden_dim 512 --batch_size 256
python3 main.py --seed 123   --lr 1e-4 --discount 0.99 --actor_hidden_dim 16 --critic_hidden_dim 512 --batch_size 256
python3 main.py --seed 1234  --lr 1e-4 --discount 0.99 --actor_hidden_dim 16 --critic_hidden_dim 512 --batch_size 256
python3 main.py --seed 12345 --lr 1e-4 --discount 0.99 --actor_hidden_dim 16 --critic_hidden_dim 512 --batch_size 256
