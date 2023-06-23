
# gym-rotor

OpenAI Gym environments for a quadrotor UAV control

<img src="https://github.com/fdcl-gwu/gym-rotor/assets/50692767/4434e07f-48ae-4d96-8407-3d815e913ca7" width=50%>

### ***Learn by Doing***

This repository contains OpenAI Gym environments and PyTorch implementations of [DDPG](https://arxiv.org/abs/1509.02971) and [TD3](https://arxiv.org/abs/1802.09477), for low-level control of quadrotor unmanned aerial vehicles. 
To better understand **What Deep RL Do**, see [OpenAI Spinning UP](https://spinningup.openai.com/en/latest/index.html).
Please don't hesitate to create new issues or pull requests for any suggestions and corrections. 
- We have recently switched from [Gym](https://www.gymlibrary.dev/) to [Gymnasium](https://gymnasium.farama.org/), but our previous Gym-based environments are still available [here](https://github.com/fdcl-gwu/gym-rotor/tree/gym).

## Installation
### Requirements
The repo was written with Python 3.11, Gymnasium 0.28.1, Pytorch 2.0.1, and Numpy 1.24.3.
It is recommended to create [Anaconda](https://www.anaconda.com/) environment with Python 3.
The official installation guide is available [here](https://docs.anaconda.com/anaconda/install/).
[Visual Studio Code](https://code.visualstudio.com/) in ``Anaconda Navigator`` is highly recommended.

1. Open your ``Anaconda Prompt`` and install major packages.
```bash
conda install -c conda-forge gymnasium
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda numpy
conda install -c conda-forge vpython
```
> Check out [Gymnasium](https://anaconda.org/conda-forge/gymnasium), [Pytorch](https://pytorch.org/get-started/locally/), and [Numpy](https://anaconda.org/anaconda/numpy), and [Vpython](https://anaconda.org/conda-forge/vpython).

2. Clone the repository.
```bash
git clone https://github.com/fdcl-gwu/gym-rotor.git
```

3. For example, training with TD3 can be run by
```bash
python main.py --env_id Quad-v0 --policy TD3
```
## Reference:
- https://github.com/openai/gym
- https://github.com/ethz-asl/reinmav-gym
- https://github.com/sfujim/TD3
