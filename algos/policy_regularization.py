import numpy as np
import torch
import torch.nn.functional as F

# Regularizing action policies for smooth control
def policy_regularization(agent_id, actor, actor_loss, batch_obs, batch_obs_next, env, args):

    # Retrieving a recent set of actions:
    if args.rl_algo == "SAC":
        batch_act = actor.sample(batch_obs)[0].clamp(-args.max_action, args.max_action)
        batch_act_next = actor.sample(batch_obs_next)[0].clamp(-args.max_action, args.max_action)
    else:
        batch_act = actor(batch_obs).clamp(-args.max_action, args.max_action)
        batch_act_next = actor(batch_obs_next).clamp(-args.max_action, args.max_action)
    
    # Temporal Smoothness:
    Loss_T = F.mse_loss(batch_act, batch_act_next)

    # Spatial Smoothness:
    noise_S = (
        torch.normal(mean=0., std=0.05, size=(1, args.obs_dim_n[agent_id]))).to(args.device) # mean and standard deviation
    if args.rl_algo == "SAC":
        batch_act_perturbed = actor.sample(batch_obs + noise_S)[0].clamp(-args.max_action, args.max_action)  # Perturbed actions
    else:
        batch_act_perturbed = actor(batch_obs + noise_S).clamp(-args.max_action, args.max_action)  # Perturbed actions
    Loss_S = F.mse_loss(batch_act, batch_act_perturbed)

    # Magnitude Smoothness:
    batch_size = batch_act.shape[0]
    if args.framework == "MONO":
        f_total_hover = np.interp(4.*env.hover_force, 
                                 [4.*env.min_force, 4.*env.max_force], 
                                 [-args.max_action, args.max_action]
                        ) * torch.ones(batch_size, 1) # normalized into [-1, 1]
        M_hover = torch.zeros(batch_size, 3)
        nominal_action = torch.cat([f_total_hover, M_hover], 1).to(args.device)
    elif args.framework == "MODUL":
        if agent_id == 0:
            f_total_hover = np.interp(4.*env.hover_force, 
                                        [4.*env.min_force, 4.*env.max_force], 
                                        [-args.max_action, args.max_action]
                                        ) * torch.ones(batch_size, 1) # normalized into [-1, 1]
            tau_hover = torch.zeros(batch_size, 3)
            nominal_action = torch.cat([f_total_hover, tau_hover], 1).to(args.device)
        elif agent_id == 1:
            nominal_action = torch.zeros(batch_size, 1).to(args.device) # M3_hover
    Loss_M = F.mse_loss(batch_act, nominal_action)

    # Regularized actor loss for smooth control:
    regularized_actor_loss = actor_loss + args.lam_T*Loss_T + args.lam_S*Loss_S + args.lam_M*Loss_M

    return regularized_actor_loss