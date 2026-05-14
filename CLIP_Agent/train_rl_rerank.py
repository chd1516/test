import argparse
from typing import Dict

import torch
import torch.nn.functional as F

from model.model import S2P
from rl_rerank import (
    ActorCriticAgent,
    QueryCandidateFuser,
    RLRerankEnv,
    RerankConfig,
    compute_gae,
    ppo_update,
)


def build_topk_candidates(query_feat: torch.Tensor, gallery_feats: torch.Tensor, top_k: int):
    query_feat = F.normalize(query_feat, dim=-1)
    gallery_feats = F.normalize(gallery_feats, dim=-1)
    scores = query_feat @ gallery_feats.t()
    k = min(top_k, gallery_feats.size(0))
    vals, ids = torch.topk(scores, k=k, dim=-1)
    return ids, vals


def train_epoch(agent, env, optimizer, query_feats, gallery_feats, gt_ids, cfg: RerankConfig):
    logs = []
    for i in range(query_feats.size(0)):
        q_feat = query_feats[i]
        cand_ids, coarse_scores = build_topk_candidates(q_feat.unsqueeze(0), gallery_feats, cfg.top_k)
        cand_ids, coarse_scores = cand_ids[0], coarse_scores[0]
        cand_feats = gallery_feats[cand_ids]
        traj, _ = env.rollout(agent, q_feat, cand_feats, cand_ids, coarse_scores, int(gt_ids[i].item()))
        adv, ret = compute_gae(traj["rewards"], traj["values"].detach(), cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        batch: Dict[str, torch.Tensor] = {
            "states": traj["states"],
            "actions": traj["actions"],
            "logp": traj["logp"],
            "advantages": adv,
            "returns": ret,
        }
        logs.append(ppo_update(agent, optimizer, batch, cfg))
    return {
        "loss": sum(x["loss"] for x in logs) / len(logs),
        "policy_loss": sum(x["policy_loss"] for x in logs) / len(logs),
        "value_loss": sum(x["value_loss"] for x in logs) / len(logs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # This script expects you to replace the toy tensors with your actual train split features.
    feat_dim, n_query, n_gallery = 512, 128, 1024
    query_feats = torch.randn(n_query, feat_dim, device=args.device)
    gallery_feats = torch.randn(n_gallery, feat_dim, device=args.device)
    gt_ids = torch.randint(0, n_gallery, (n_query,), device=args.device)

    cfg = RerankConfig(top_k=args.top_k)
    fuser = QueryCandidateFuser(feat_dim).to(args.device).eval()
    agent = ActorCriticAgent(feat_dim=feat_dim).to(args.device)
    env = RLRerankEnv(fuser, cfg)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        stats = train_epoch(agent, env, optimizer, query_feats, gallery_feats, gt_ids, cfg)
        print(f"epoch={epoch} stats={stats}")

    torch.save({"agent": agent.state_dict(), "cfg": cfg.__dict__}, "rl_rerank_agent.pth")


if __name__ == "__main__":
    main()
