
# gym-rotor

OpenAI Gym environments for a quadrotor UAV control

<img src="https://github.com/fdcl-gwu/gym-rotor/assets/50692767/4434e07f-48ae-4d96-8407-3d815e913ca7" width=50%>

### ***Learn by Doing***

This repository contains OpenAI Gym environments and PyTorch implementations of [TD3](https://arxiv.org/abs/1802.09477) and [MATD3](https://arxiv.org/abs/1910.01465), for low-level control of quadrotor unmanned aerial vehicles. 
To better understand **What Deep RL Do**, see [OpenAI Spinning UP](https://spinningup.openai.com/en/latest/index.html).
Please don't hesitate to create new issues or pull requests for any suggestions and corrections. 
- We have recently switched from [Gym](https://www.gymlibrary.dev/) to [Gymnasium](https://gymnasium.farama.org/), but our previous Gym-based environments are still available [here](https://github.com/fdcl-gwu/gym-rotor/tree/gym).

## Installation
### Requirements
The repo was written with Python 3.11.3, Gymnasium 0.28.1, Pytorch 2.0.1, and Numpy 1.25.1.
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
Consider a quadrotor UAV below. The equations of motion are given by

<img src="https://github.com/fdcl-gwu/gym-rotor/assets/50692767/7d683754-fd60-41e0-a29f-12e26ea279a8" width=40%>

The position and the velocity of the quadrotor are represented by $x \in \mathbb{R}^3$ and $v \in \mathbb{R}^3$, respectively.
The attitude is defined by the rotation matrix $R \in SO(3) = \lbrace R \in \mathbb{R}^{3\times 3} | R^T R=I_{3\times 3}, \mathrm{det}[R]=1 \rbrace$, that is the linear transformation of the representation of a vector from the body-fixed frame $\lbrace \vec b_{1},\vec b_{2},\vec b_{3} \rbrace$ to the inertial frame $\lbrace \vec e_1,\vec e_2,\vec e_3 \rbrace$. 
The angular velocity vector is denoted by $\Omega \in \mathbb{R}^3$.
Given the total thrust $f = \sum{}_{i=1}^{4} T_i \in \mathbb{R}$ and the moment $M = [M_1, M_2, M_3]^T \in \mathbb{R}^3$ resolved in the body-fixed frame, the thrust of each motor $(T_1,T_2,T_3,T_4)$ is determined by

$$ \begin{gather} 
    \begin{bmatrix} 
        T_1 \\\ T_2 \\\ T_3 \\\ T_4
    \end{bmatrix}
    = \frac{1}{4}
    \begin{bmatrix}
        1 & 0      & 2/d   & -1/c_{\tau f} \\
        1 & -2/d & 0      & 1/c_{\tau f} \\
        1 & 0      & -2/d & -1/c_{\tau f} \\
        1 & 2/d   & 0      & 1/c_{\tau f} 
    \end{bmatrix}
    \begin{bmatrix}
        f \\\ M_1 \\\ M_2 \\\ M_3 
    \end{bmatrix}.
\end{gather} $$

| Env IDs | Description |
| :---: | --- |
| `Quad-v0` | This serves as the foundational env for wrappers, where the state and action are represented as $s = (x, v, R, \Omega)$ and $a = (T_1, T_2, T_3, T_4)$.|
| `CoupledWrapper` | For single-agent RL frameworks; the observation and action are given by $o = (e_x, e_v, R, e_\Omega, e_{I_x}, e_{b_1}, e_{I_{b_1}})$ and $a = (f, M_1, M_2, M_3)$.|
| `DecoupledWrapper` | For multi-agent RL frameworks; the observation and action for each agent are defined as $o_1 = (e_x, e_v, b_3, e_{\omega_{12}}, e_{I_x})$, $a_1 = (f, \tau)$ and $o_2 = (b_1, e_{\Omega_3}, e_{b_1}, e_{I_{b_1}})$, $a_2 = M_3$, respectively.|

where the error terms $e_x, e_v$, and $e_\Omega$ represent the errors in position, velocity, and angular velocity, respectively.
To eliminate steady-state errors, we add the integral terms $e_{I_x}$ and $e_{I_{b_1}}$.
More details can be found [here](https://arxiv.org/abs/2311.06144).

<!-- ### wrapper
This repo provides several useful wrappers that can be found in `./gym_rotor/wrappers/'.
| Wrapper IDs | Description |
| :---: | --- |
| `Sim2RealWrapper` | [Domain randomization](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/) and sensor noise are modeled for sim-to-real transfer.|
| `EquivWrapper` | Rotation equivariance properties are implemented for sample efficiency. More details can be found [here](https://arxiv.org/abs/2206.01233).| -->

## Examples
Hyperparameters can be adjusted in `args_parse.py`.
For example, training with the CTDE framework can be run by
```bash
python3 main.py --framework CTDE --seed 789
```

## Citation
If you find this work useful in your own work or would like to cite it, please give credit to our work:
```bash
@article{yu2023multi,
  title={Multi-Agent Reinforcement Learning for the Low-Level Control of a Quadrotor UAV},
  author={Yu, Beomyeol and Lee, Taeyoung},
  journal={arXiv preprint arXiv:2311.06144},
  year={2023}
}

@inproceedings{yu2023equivariant,
  title={Equivariant Reinforcement Learning for Quadrotor UAV},
  author={Yu, Beomyeol and Lee, Taeyoung},
  booktitle={2023 American Control Conference (ACC)},
  pages={2842--2847},
  year={2023},
  organization={IEEE}
}
```

## Reference:
- https://github.com/ethz-asl/reinmav-gym
- https://github.com/Lizhi-sjtu/MARL-code-pytorch