"""
CEM Planner - Searches over hybrid action space (continuous + discrete).
"""

import torch
from typing import Tuple


class CEMPlanner:
    """
    Cross-Entropy Method planner for hybrid action space.
    
    Searches over:
    - Continuous actions: cursor velocity/position (2D)
    - Discrete actions: generic action tokens (0 = NO-OP, 1..N-1 = env-specific executed actions)
    """
    
    def __init__(
        self,
        continuous_dim: int = 2,
        num_discrete_actions: int = 11,
        horizon: int = 10,
        num_samples: int = 100,
        num_elites: int = 10,
        num_iterations: int = 5,
        momentum: float = 0.1,
    ):
        self.continuous_dim = continuous_dim
        self.num_discrete_actions = num_discrete_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.momentum = momentum
        self.cont_mean = None
        self.cont_std = None
        self.disc_probs = None
        
    def reset(self):
        self.cont_mean = None
        self.cont_std = None
        self.disc_probs = None
        
    @torch.no_grad()
    def plan(self, z_current, world_model, reward_fn, device=None):
        if device is None:
            device = z_current.device
        z_current = z_current.unsqueeze(0) if z_current.dim() == 1 else z_current
        
        if self.cont_mean is None:
            self.cont_mean = torch.zeros(self.horizon, self.continuous_dim, device=device)
            self.cont_std = torch.ones(self.horizon, self.continuous_dim, device=device) * 0.5
            self.disc_probs = torch.ones(self.horizon, self.num_discrete_actions, device=device) / self.num_discrete_actions
        
        cont_mean, cont_std = self.cont_mean.clone(), self.cont_std.clone()
        disc_probs = self.disc_probs.clone()
        best_cont, best_disc, best_reward = None, 0, float("-inf")
        
        for _ in range(self.num_iterations):
            noise = torch.randn(self.num_samples, self.horizon, self.continuous_dim, device=device)
            cont_actions = torch.clamp(cont_mean + cont_std * noise, -1.0, 1.0)
            
            disc_actions = torch.zeros(self.num_samples, self.horizon, dtype=torch.long, device=device)
            for t in range(self.horizon):
                disc_actions[:, t] = torch.multinomial(disc_probs[t].expand(self.num_samples, -1), 1).squeeze()
            
            z_batch = z_current.expand(self.num_samples, -1)
            z_traj = world_model.multi_step_rollout(z_batch, cont_actions, disc_actions)
            
            # Compute rewards - pass action trajectory for reward prediction
            rewards_list = []
            for i in range(self.num_samples):
                # Try to pass actions to reward_fn for step-reward prediction
                try:
                    r = reward_fn(z_traj[i], actions_traj=(cont_actions[i], disc_actions[i]))
                except TypeError:
                    # Fallback for reward functions that don't accept actions
                    r = reward_fn(z_traj[i])
                rewards_list.append(r)
            rewards = torch.tensor(rewards_list, device=device)
            
            elite_idxs = torch.argsort(rewards, descending=True)[:self.num_elites]
            elite_cont, elite_disc = cont_actions[elite_idxs], disc_actions[elite_idxs]
            
            cont_mean = self.momentum * cont_mean + (1 - self.momentum) * elite_cont.mean(0)
            cont_std = self.momentum * cont_std + (1 - self.momentum) * (elite_cont.std(0) + 1e-6)
            
            new_probs = torch.zeros_like(disc_probs)
            for t in range(self.horizon):
                for a in range(self.num_discrete_actions):
                    new_probs[t, a] = (elite_disc[:, t] == a).float().sum() + 1e-6
                new_probs[t] /= new_probs[t].sum()
            disc_probs = self.momentum * disc_probs + (1 - self.momentum) * new_probs
            
            if rewards[elite_idxs[0]] > best_reward:
                best_reward = float(rewards[elite_idxs[0]])
                best_cont = cont_actions[elite_idxs[0], 0].clone()
                best_disc = int(disc_actions[elite_idxs[0], 0].item())
        
        self.cont_mean = torch.cat([cont_mean[1:], torch.zeros(1, self.continuous_dim, device=device)])
        self.cont_std = torch.cat([cont_std[1:], torch.ones(1, self.continuous_dim, device=device) * 0.5])
        self.disc_probs = torch.cat([disc_probs[1:], torch.ones(1, self.num_discrete_actions, device=device) / self.num_discrete_actions])
        
        return best_cont, best_disc, best_reward


class MPPIPlanner:
    """MPPI planner for hybrid actions."""
    
    def __init__(self, continuous_dim=2, num_discrete_actions=9, horizon=10, num_samples=100, temperature=1.0):
        self.continuous_dim = continuous_dim
        self.num_discrete_actions = num_discrete_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.cont_mean = None
        self.disc_probs = None
        
    def reset(self):
        self.cont_mean = None
        self.disc_probs = None
        
    @torch.no_grad()
    def plan(self, z_current, world_model, reward_fn, device=None):
        if device is None:
            device = z_current.device
        z_current = z_current.unsqueeze(0) if z_current.dim() == 1 else z_current
        
        if self.cont_mean is None:
            self.cont_mean = torch.zeros(self.horizon, self.continuous_dim, device=device)
            self.disc_probs = torch.ones(self.horizon, self.num_discrete_actions, device=device) / self.num_discrete_actions
        
        noise = torch.randn(self.num_samples, self.horizon, self.continuous_dim, device=device) * 0.3
        cont_actions = torch.clamp(self.cont_mean + noise, -1.0, 1.0)
        
        disc_actions = torch.zeros(self.num_samples, self.horizon, dtype=torch.long, device=device)
        for t in range(self.horizon):
            disc_actions[:, t] = torch.multinomial(self.disc_probs[t].expand(self.num_samples, -1), 1).squeeze()
        
        z_batch = z_current.expand(self.num_samples, -1)
        z_traj = world_model.multi_step_rollout(z_batch, cont_actions, disc_actions)
        rewards = torch.tensor([reward_fn(z_traj[i]) for i in range(self.num_samples)], device=device)
        
        weights = torch.softmax(rewards / self.temperature, dim=0)
        weighted_cont = (weights.view(-1, 1, 1) * cont_actions).sum(0)
        
        weighted_probs = torch.zeros(self.horizon, self.num_discrete_actions, device=device)
        for i in range(self.num_samples):
            for t in range(self.horizon):
                weighted_probs[t, disc_actions[i, t]] += weights[i]
        weighted_probs /= weighted_probs.sum(1, keepdim=True)
        
        best_cont = weighted_cont[0]
        best_disc = int(weighted_probs[0].argmax().item())
        
        self.cont_mean = torch.cat([weighted_cont[1:], torch.zeros(1, self.continuous_dim, device=device)])
        self.disc_probs = torch.cat([weighted_probs[1:], torch.ones(1, self.num_discrete_actions, device=device) / self.num_discrete_actions])
        
        return best_cont, best_disc, float(rewards.max())
