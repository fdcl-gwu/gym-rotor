from gymnasium.envs.registration import registry, register, make, spec

register(
    id='Quad-v0',
    entry_point='gym_rotor.envs:QuadEnv',
    max_episode_steps = 10000,
)