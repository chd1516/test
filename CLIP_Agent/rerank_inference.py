import torch
import torch.nn.functional as F

from rl_rerank import ACTION_FUSE, ACTION_NEXT, ACTION_STOP, ActorCriticAgent, QueryCandidateFuser, RLRerankEnv, RerankConfig


@torch.no_grad()
def rerank_once(agent, env, query_feat, gallery_feats, top_k=64):
    q = F.normalize(query_feat, dim=-1)
    g = F.normalize(gallery_feats, dim=-1)
    scores = q @ g.t()
    vals, ids = torch.topk(scores, k=min(top_k, g.size(0)), dim=-1)
    cand_ids = ids[0]
    cand_feats = g[cand_ids]
    coarse = vals[0]
    traj, best_id = env.rollout(agent, q[0], cand_feats, cand_ids, coarse, gt_id=-1)
    return best_id, cand_ids, coarse, traj["actions"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = 512
    cfg = RerankConfig(top_k=32)
    fuser = QueryCandidateFuser(feat_dim).to(device).eval()
    agent = ActorCriticAgent(feat_dim=feat_dim).to(device)
    env = RLRerankEnv(fuser, cfg)

    query = torch.randn(1, feat_dim, device=device)
    gallery = torch.randn(1000, feat_dim, device=device)
    best_id, _, _, actions = rerank_once(agent, env, query, gallery, top_k=32)
    print("best_id=", best_id)
    print("actions=", actions.cpu().tolist())
