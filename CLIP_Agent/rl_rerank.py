from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


ACTION_NEXT = 0
ACTION_FUSE = 1
ACTION_STOP = 2
ACTION_NAMES = {ACTION_NEXT: "NEXT", ACTION_FUSE: "FUSE", ACTION_STOP: "STOP"}


@dataclass
class RerankConfig:
    top_k: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_steps: int = 64
    fuse_cost: float = -0.01
    fuse_hit_reward: float = 0.1
    stop_success_reward: float = 1.0
    stop_fail_penalty: float = -1.0


class QueryCandidateFuser(nn.Module):
    """A light fusion scorer as placeholder for expensive cross-encoder FUSE."""

    def __init__(self, feat_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 3 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q_feat: torch.Tensor, c_feat: torch.Tensor, coarse_score: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([q_feat, c_feat, q_feat * c_feat, coarse_score, (q_feat - c_feat).norm(dim=-1, keepdim=True)], dim=-1)
        return self.net(pair).squeeze(-1)


class StateEncoder(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = feat_dim * 2 + 7
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        y, h = self.rnn(x.unsqueeze(1), hx)
        return y[:, -1], h


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = StateEncoder(feat_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, 3)
        self.critic = nn.Linear(hidden_dim, 1)

    def step(self, state_vec: torch.Tensor, hx: Optional[torch.Tensor] = None):
        h, new_hx = self.encoder(state_vec, hx)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, new_hx, logits

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        h, _ = self.encoder(states, None)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        value = self.critic(h).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), value


class RLRerankEnv:
    def __init__(self, fuser: nn.Module, cfg: RerankConfig):
        self.fuser = fuser
        self.cfg = cfg

    def _build_state(
        self,
        q_feat: torch.Tensor,
        cand_feats: torch.Tensor,
        coarse_scores: torch.Tensor,
        ptr: int,
        used_fuse: int,
        best_fine: float,
        second_fine: float,
    ) -> torch.Tensor:
        k = cand_feats.shape[0]
        cur_feat = cand_feats[min(ptr, k - 1)]
        cur_coarse = coarse_scores[min(ptr, k - 1)]
        margin = best_fine - second_fine if second_fine > -1e8 else 0.0
        stats = torch.tensor([
            ptr / max(k, 1),
            float(cur_coarse),
            float(best_fine if best_fine > -1e8 else 0.0),
            float(margin),
            float((k - ptr) / max(k, 1)),
            float(used_fuse / max(k, 1)),
            1.0 if best_fine > -1e8 else 0.0,
        ], device=q_feat.device, dtype=q_feat.dtype)
        return torch.cat([q_feat, cur_feat, stats], dim=-1).unsqueeze(0)

    @torch.no_grad()
    def rollout(self, agent: ActorCriticAgent, q_feat: torch.Tensor, cand_feats: torch.Tensor, cand_ids: torch.Tensor,
                coarse_scores: torch.Tensor, gt_id: int):
        ptr, used_fuse = 0, 0
        best_fine, second_fine, best_id = -1e9, -1e9, int(cand_ids[0].item())
        done, step = False, 0
        traj: Dict[str, List[torch.Tensor]] = {k: [] for k in ["states", "actions", "logp", "entropy", "values", "rewards"]}
        hx = None

        while not done and step < self.cfg.max_steps and ptr < cand_feats.shape[0]:
            state = self._build_state(q_feat, cand_feats, coarse_scores, ptr, used_fuse, best_fine, second_fine)
            action, logp, entropy, value, hx, _ = agent.step(state, hx)
            a = int(action.item())
            reward = 0.0

            if a == ACTION_NEXT:
                ptr += 1
                if ptr >= cand_feats.shape[0]:
                    done = True
            elif a == ACTION_FUSE:
                fine = float(self.fuser(q_feat.unsqueeze(0), cand_feats[ptr].unsqueeze(0), coarse_scores[ptr].view(1, 1)).item())
                used_fuse += 1
                reward += self.cfg.fuse_cost
                if int(cand_ids[ptr].item()) == gt_id:
                    reward += self.cfg.fuse_hit_reward
                if fine > best_fine:
                    second_fine, best_fine = best_fine, fine
                    best_id = int(cand_ids[ptr].item())
                elif fine > second_fine:
                    second_fine = fine
                ptr += 1
                if ptr >= cand_feats.shape[0]:
                    done = True
            else:
                done = True

            if done:
                reward += self.cfg.stop_success_reward if best_id == gt_id else self.cfg.stop_fail_penalty

            for k, v in {
                "states": state.squeeze(0),
                "actions": action,
                "logp": logp,
                "entropy": entropy,
                "values": value,
                "rewards": torch.tensor(reward, device=state.device),
            }.items():
                traj[k].append(v)
            step += 1

        return {k: torch.stack(v) for k, v in traj.items()}, best_id


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float):
    adv = torch.zeros_like(rewards)
    lastgaelam = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
        next_value = values[t]
    returns = adv + values
    return adv, returns


def ppo_update(agent: ActorCriticAgent, optimizer: torch.optim.Optimizer, batch: Dict[str, torch.Tensor], cfg: RerankConfig):
    states, actions = batch["states"], batch["actions"]
    old_logp, advantages, returns = batch["logp"].detach(), batch["advantages"].detach(), batch["returns"].detach()
    new_logp, entropy, values = agent.evaluate_actions(states, actions)
    ratio = (new_logp - old_logp).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = (returns - values).pow(2).mean()
    entropy_loss = -entropy.mean()
    loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item()), "policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item())}
