
# gym-rotor

OpenAI Gym environments for a quadrotor UAV control

### ***Learn by Doing***

This repository contains OpenAI Gym environments and PyTorch implementations of [DDPG](https://arxiv.org/abs/1509.02971) and [TD3](https://arxiv.org/abs/1802.09477), for low-level control of quadrotor unmanned aerial vehicles. 
To better understand **What Deep RL Do**, see [OpenAI Spinning UP](https://spinningup.openai.com/en/latest/index.html).
Please don't hesitate to create new issues or pull requests for any suggestions and corrections. 
- We have recently switched from [Gym](https://www.gymlibrary.dev/) to [Gymnasium](https://gymnasium.farama.org/), but our previous Gym-based environments are still available [here](https://github.com/fdcl-gwu/gym-rotor/tree/gym).

## Installation
### Requirements
The repo was written using Python 3.7 with conda on macOS 10.15 and tested with Python 3.8 on macOS 12, Ubuntu 20.04

It is recommended to create [Anaconda](https://www.anaconda.com/) environment with Python 3.
The official installation guide is available [here](https://docs.anaconda.com/anaconda/install/).
[Visual Studio Code](https://code.visualstudio.com/) in ``Anaconda Navigator`` is highly recommended.

1. Open your ``Anaconda Prompt`` and install major packages.
```bash
conda install -c conda-forge gym 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
conda install -c conda-forge vpython
```
> Check out [Gym](https://anaconda.org/conda-forge/gym), [Pytorch](https://pytorch.org/), and [Vpython](https://anaconda.org/conda-forge/vpython).

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
