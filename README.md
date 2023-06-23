
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

## Environments
Consider a quadrotor UAV below:

<img src="https://github.com/fdcl-gwu/gym-rotor/assets/50692767/7d683754-fd60-41e0-a29f-12e26ea279a8" width=40%>

The position and the velocity of the quadrotor are represented by $x \in \mathbb{R}^3$ and $v \in \mathbb{R}^3$, respectively.
The attitude is defined by the rotation matrix $R \in SO(3) = \lbrace R \in \mathbb{R}^{3\times 3} | R^T R=I_{3\times 3}, \mathrm{det}[R]=1 \rbrace$, that is the linear transformation of the representation of a vector from the body-fixed frame $\lbrace \vec b_{1},\vec b_{2},\vec b_{3} \rbrace$ to the inertial frame $\lbrace \vec e_1,\vec e_2,\vec e_3 \rbrace$. 
The angular velocity vector is denoted by $\Omega \in \mathbb{R}^3$.
From the thrust of each motor $(T_1,T_2,T_3,T_4)$, the total thrust $f = \sum{}_{i=1}^{4} T_i \in \mathbb{R}$ and the total moment $M \in \mathbb{R}^3$ resolved in the body-fixed frame can be represented.

| Env IDs | Description |
| :---: | --- |
| `Quad-v0` | The state and the action are given by $s = (e_x, e_v, R, e_\Omega)$ and $a = (T_1, T_2, T_3, T_4)$.|
| `Quad-v1` | The state and the action are given by $s = (e_x, eI_x, e_v, R, e_\Omega)$ and $a = (T_1, T_2, T_3, T_4)$.|

where the error terms $e_x, e_v$, and $e_\Omega$ represent the errors in position, velocity, and angular velocity, respectively.
Note that `Quad-v0` often suffers from a problem with steady-state errors in position, so we add an integral term $eI_x$ to `Quad-v1` to address this issue.

### wrapper
This repo provides several useful wrappers that can be found in `./gym_rotor/wrappers/'.
| Wrapper IDs | Description |
| :---: | --- |
| `Sim2RealWrapper` | [Domain randomization](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/) and sensor noise are modeled for sim-to-real transfer.|
| `EquivWrapper` | Rotation equivariance properties are implemented for sample efficiency. More details can be found [here](https://arxiv.org/abs/2206.01233).|

## Examples
Hyperparameters can be adjusted in `args_parse.py`.
For example, training with TD3 can be run by
```bash
python main.py --env_id Quad-v1 --policy TD3
```

## Citation
If you find this work useful in your own work or would like to cite it, please give credit to our work:
```bash
@article{yu2022equivariant,
  title={Equivariant Reinforcement Learning for Quadrotor UAV},
  author={Yu, Beomyeol and Lee, Taeyoung},
  journal={arXiv preprint arXiv:2206.01233},
  year={2022}
}
```

## Reference:
- https://github.com/openai/gym
- https://github.com/ethz-asl/reinmav-gym
- https://github.com/sfujim/TD3
