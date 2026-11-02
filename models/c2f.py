"""Coarse-to-fine trajectory world models: concept-first, continuous NCM, and baselines.

One model class covers the method and its baselines, so they share the same
transformer backbone and training code:

- levels=[H]                       -> flat flow matching (CFM, x1-prediction)
- levels=[H], mode="diff"          -> flat diffusion (VP cosine schedule, x0-prediction,
                                      DDIM / ancestral sampling)
- levels=[H], mode="reg"           -> deterministic regression
- levels=[1, 2, 4, ..., H]         -> continuous NCM: coarse-to-fine residual levels
- concept_k=K                      -> concept-first: the first level is a discrete choice
                                      among K learned coarse futures ("concepts", k-means
                                      codebook), with an exact categorical distribution;
                                      the finer levels are continuous residual flows
- residual=False                   -> each level predicts the full downsampled signal
                                      (HiFlow-style ablation)
- corrupt=0                        -> no corruption of the coarser levels (ablation)

Scales follow NCM (arXiv 2610.14162): area pooling down, linear interpolation up,
residuals between levels, block-diagonal attention plus a shared prefix, and
corruption of the conditioning levels (NCM: a fraction of conditioning codes is replaced
by random codes of the same level, at a rate decreasing linearly across levels with mean
0.5, so the generator tolerates its own errors at inference). Here a continuous token is
replaced by a draw from N(0, 1) in per-level std units, at rates falling linearly from
1.5*corrupt to 0.5*corrupt over the levels that are conditioned on (the last level never
is). A discrete concept is replaced by a uniformly random concept at its own, smaller
rate (corrupt_concept): it is sampled from a calibrated categorical at inference, so a
sampled concept is not an error the finer levels should learn to ignore.

Sampling note: with x1-prediction, the first step starts from pure noise, which carries
no information about the target, so an ideal network predicts the conditional mean there.
A continuous level therefore needs several steps (nfe > 1) to produce diverse samples;
nfe = 1 collapses towards the mean. A discrete level is diverse in a single pass.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM, SCAN_DIM, PLAN_DIM = 46, 49, 10
HIST, PAR_DIM, MAP_DIM = 25, 11, 289


_MATS = {}


def _resize_matrix(kind, src, dst, device):
    """Time-resizing as a (dst, src) matrix, built once on the CPU with the torch
    reference ops (area pooling / linear interpolation) and cached per device.
    A matmul works on every backend; MPS lacks non-divisible adaptive pooling."""
    key = (kind, src, dst, str(device))
    if key not in _MATS:
        eye = torch.eye(src).unsqueeze(0)                       # (1, src, src) as channels
        if kind == "down":
            m = F.adaptive_avg_pool1d(eye, dst)[0]              # (src, dst)
        else:
            m = F.interpolate(eye, size=dst, mode="linear", align_corners=False)[0]
        _MATS[key] = m.t().contiguous().to(device)              # (dst, src)
    return _MATS[key]


def down(x, length):
    """(B, L, D) -> (B, length, D) by area pooling over time."""
    if x.shape[1] == length:
        return x
    return _resize_matrix("down", x.shape[1], length, x.device) @ x


def up(x, length):
    """(B, l, D) -> (B, length, D) by linear interpolation over time."""
    if x.shape[1] == length:
        return x
    return _resize_matrix("up", x.shape[1], length, x.device) @ x


def decompose(x, levels, residual=True, recon=None):
    """Ground-truth level targets r_k (B, l_k, D) for a trajectory x (B, H, D),
    starting from an optional reconstruction (e.g. a concept's coarse future)."""
    H = x.shape[1]
    rs, recon = [], torch.zeros_like(x) if recon is None else recon
    for l in levels:
        r = down(x - recon, l) if residual else down(x, l)
        rs.append(r)
        recon = recon + up(r, H) if residual else up(r, H)
    return rs


def sinusoid(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


def alpha_bar(t):
    """VP cosine schedule in 'data time' t (0 = pure noise, 1 = clean)."""
    return torch.sin(0.5 * math.pi * t.clamp(0, 1)) ** 2


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.h = heads

    def forward(self, x, mask):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))


class C2FWorldModel(nn.Module):
    def __init__(self, horizon, levels=None, d=256, depth=4, heads=4, residual=True,
                 corrupt=0.5, alpha=0.5, mode="flow", concept_k=0, corrupt_concept=0.1):
        super().__init__()
        self.H = horizon
        self.levels = list(levels or [horizon])
        assert self.levels[-1] == horizon
        assert not (concept_k and len(self.levels) < 2), "concept-first needs >= 2 levels"
        self.residual, self.corrupt, self.alpha, self.mode = residual, corrupt, alpha, mode
        self.K = concept_k
        self.corrupt_concept = corrupt_concept
        self.d = d
        self.in_x = nn.Linear(STATE_DIM, d)          # noisy level tokens
        self.in_ctx = nn.Linear(STATE_DIM, d)        # resized reconstruction so far
        self.in_plan = nn.Linear(PLAN_DIM, d)
        self.in_s0 = nn.Linear(STATE_DIM, d)
        self.in_hist = nn.Linear(STATE_DIM * HIST, d)
        self.in_scan = nn.Linear(SCAN_DIM, d)
        self.in_par = nn.Linear(PAR_DIM, d)
        self.in_map = nn.Sequential(nn.Linear(MAP_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.level_emb = nn.Embedding(len(self.levels) + 1, d)
        self.t_mlp = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, STATE_DIM)
        if self.K:
            self.concept_head = nn.Linear(d, self.K)
            self.register_buffer("codebook", torch.zeros(self.K, self.levels[0], STATE_DIM))
            self.register_buffer("code_fall", torch.zeros(self.K))   # P(fall | concept)
        # per-level residual scale, set from data before training (normalized units)
        self.register_buffer("lvl_std", torch.ones(len(self.levels), STATE_DIM))
        w = torch.tensor([l ** (1 - alpha) for l in self.levels], dtype=torch.float32)
        if self.K:
            w[0] = w[1:].mean()                       # CE and MSE terms on a similar footing
        self.register_buffer("lvl_w", w / w.sum())

    # ---------------------------------------------------------------- data fitting
    @torch.no_grad()
    def fit_level_std(self, x):
        rs = decompose(x, self.levels, self.residual)
        if self.K:
            self.lvl_std[0] = rs[0].reshape(-1, STATE_DIM).std(0).clamp_min(1e-3)
            recon = up(self.codebook[self.assign(rs[0])] * self.lvl_std[0], self.H) \
                if self.codebook.abs().sum() > 0 else up(rs[0], self.H)
            rs = rs[:1] + decompose(x, self.levels[1:], self.residual, recon)
        for k, r in enumerate(rs):
            self.lvl_std[k] = r.reshape(-1, STATE_DIM).std(0).clamp_min(1e-3)

    @torch.no_grad()
    def fit_concepts(self, x, fell, iters=50, seed=0):
        """k-means (k-means++ init) on the coarsest level, in per-level std units.
        x: (N, H, D) normalized trajectories; fell: (N,) bool fall-within-horizon labels."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        c = down(x, self.levels[0])
        self.lvl_std[0] = c.reshape(-1, STATE_DIM).std(0).clamp_min(1e-3)
        z = (c / self.lvl_std[0]).reshape(len(c), -1)
        cent = z[torch.randint(len(z), (1,), generator=g)]
        for _ in range(1, self.K):                      # k-means++
            d2 = torch.cdist(z, cent).min(1).values ** 2
            cent = torch.cat([cent, z[torch.multinomial(d2.cpu() / d2.sum().cpu(), 1,
                                                         generator=g).to(z.device)]])
        for _ in range(iters):
            a = torch.cdist(z, cent).argmin(1)
            for k in range(self.K):
                sel = a == k
                if sel.any():
                    cent[k] = z[sel].mean(0)
                else:                                   # re-seed empty clusters
                    cent[k] = z[torch.randint(len(z), (1,), generator=g)]
        self.codebook.copy_(cent.view(self.K, self.levels[0], STATE_DIM))
        a = self.assign(c / self.lvl_std[0])
        f = fell.float().to(z.device)
        cnt = torch.bincount(a, minlength=self.K).float()
        self.code_fall.copy_((torch.bincount(a, weights=f, minlength=self.K) + 0.5) / (cnt + 1.0))
        return cnt

    def assign(self, c_n):
        """(B, l0, D) coarse level in std units -> nearest concept id (B,)."""
        return torch.cdist(c_n.reshape(len(c_n), -1),
                           self.codebook.reshape(self.K, -1)).argmin(1)

    # ---------------------------------------------------------------- network
    def _prefix(self, b):
        p = torch.stack([self.in_s0(b["s0"]), self.in_hist(b["hist"].flatten(1)),
                         self.in_scan(b["scan"]), self.in_map(b["tmap"]), self.in_par(b["par"])], 1)
        return p + self.level_emb.weight[0]

    def _block_tokens(self, k, x_noisy, ctx_full, plan, t):
        l = self.levels[k]
        pos = (torch.arange(l, device=x_noisy.device) + 0.5) / l
        tok = (self.in_x(x_noisy) + self.in_ctx(down(ctx_full, l)) + self.in_plan(down(plan, l))
               + self.level_emb.weight[k + 1] + sinusoid(pos * 50, self.d)[None])
        return tok + self.t_mlp(sinusoid(t * 100, 64))[:, None]

    def _run(self, prefix, blocks, raw=False):
        """prefix (B, P, d); blocks: list of (B, l_k, d). Block-diagonal attention:
        each block sees the prefix and itself; the prefix sees only itself."""
        sizes = [prefix.shape[1]] + [b.shape[1] for b in blocks]
        dev = prefix.device
        ids = torch.repeat_interleave(torch.arange(len(sizes), device=dev),
                                      torch.tensor(sizes, device=dev))
        mask = (ids[:, None] == ids[None, :]) | (ids[None, :] == 0)
        h = torch.cat([prefix] + blocks, 1)
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.norm(h)
        out, i = [], sizes[0]
        for s in sizes[1:]:
            out.append(h[:, i:i + s] if raw else self.out(h[:, i:i + s]))
            i += s
        return out

    def _noisy(self, r, t):
        """Corrupt clean level r (normalized) to data time t for flow or diffusion."""
        e = torch.randn_like(r)
        tt = t[:, None, None]
        if self.mode == "diff":
            ab = alpha_bar(tt)
            return ab.sqrt() * r + (1 - ab).sqrt() * e
        return (1 - tt) * e + tt * r

    # ---------------------------------------------------------------- training
    def corrupt_rates(self, first):
        """Corruption rate per continuous conditioning level (all but the last level),
        linear from 1.5*corrupt (coarsest) to 0.5*corrupt (finest), mean = corrupt."""
        cond = list(range(first, len(self.levels) - 1))
        n = len(cond)
        return {k: self.corrupt * ((1.5 - j / (n - 1)) if n > 1 else 1.0)
                for j, k in enumerate(cond)}

    def loss(self, batch):
        plan, x = batch["plan"], batch["x"]
        B, H = x.shape[:2]
        dev = x.device
        K = len(self.levels)
        ctxs, recon = [], torch.zeros_like(x)
        if self.K:
            c0_n = down(x, self.levels[0]) / self.lvl_std[0]
            code = self.assign(c0_n)
            ctxs.append(recon)
            p0 = self.corrupt_concept if self.training else 0.0
            code_in = torch.where(torch.rand(B, device=dev) < p0,
                                  torch.randint(self.K, (B,), device=dev), code)
            recon = up(self.codebook[code_in] * self.lvl_std[0], H)
            rs = [None] + decompose(x, self.levels[1:], self.residual,
                                    up(self.codebook[code] * self.lvl_std[0], H))
            first = 1
        else:
            rs = decompose(x, self.levels, self.residual)
            first = 0
        rs_n = [None if r is None else r / self.lvl_std[k] for k, r in enumerate(rs)]
        # corrupted conditioning: tokens of coarser levels replaced by random values
        # with probability falling linearly from 2*corrupt (level 1) to 0 (last level)
        rates = self.corrupt_rates(first)
        for k in range(first, K):
            ctxs.append(recon)
            p = rates.get(k, 0.0)
            r = rs_n[k]
            if p > 0 and self.training:
                m = (torch.rand(B, r.shape[1], 1, device=dev) < p).float()
                r = r * (1 - m) + torch.randn_like(r) * m
            r = r * self.lvl_std[k]
            recon = recon + up(r, H) if self.residual else up(r, H)
        blocks, targets = [], []
        for k in range(K):
            l = self.levels[k]
            if k < first or self.mode == "reg":
                t = torch.ones(B, device=dev)
                xin = torch.zeros(B, l, STATE_DIM, device=dev)
            else:
                t = torch.rand(B, device=dev)
                xin = self._noisy(rs_n[k], t)
            blocks.append(self._block_tokens(k, xin, ctxs[k], plan, t))
            targets.append(rs_n[k])
        outs = self._run(self._prefix(batch), blocks, raw=True)
        per_level = []
        for k in range(K):
            if k < first:
                logits = self.concept_head(outs[0].mean(1))
                per_level.append(F.cross_entropy(logits, code))
            else:
                per_level.append(F.mse_loss(self.out(outs[k]), targets[k]))
        per_level = torch.stack(per_level)
        return (per_level * self.lvl_w).sum(), per_level.detach()

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def concept_probs(self, batch, temperature=1.0):
        B = batch["s0"].shape[0]
        dev = batch["s0"].device
        tok = self._block_tokens(0, torch.zeros(B, self.levels[0], STATE_DIM, device=dev),
                                 torch.zeros(B, self.H, STATE_DIM, device=dev), batch["plan"],
                                 torch.ones(B, device=dev))
        h = self._run(self._prefix(batch), [tok], raw=True)[0]
        return F.softmax(self.concept_head(h.mean(1)) / temperature, -1)

    @torch.no_grad()
    def sample(self, batch, nfe=1, temperature=1.0, eta=0.0, codes=None, return_levels=False):
        """batch: model inputs (s0, hist, scan, par, plan). nfe: network passes per
        continuous level. eta: 0 = deterministic (DDIM / Euler), 1 = ancestral (diffusion).
        codes: optional fixed concept ids (B,) for concept-first models."""
        B = batch["s0"].shape[0]
        dev = batch["s0"].device
        prefix = self._prefix(batch)
        plan = batch["plan"]
        recon = torch.zeros(B, self.H, STATE_DIM, device=dev)
        levels_out, first = [], 0
        if self.K:
            if codes is None:
                codes = torch.multinomial(self.concept_probs(batch, temperature), 1)[:, 0]
            recon = up(self.codebook[codes] * self.lvl_std[0], self.H)
            levels_out.append(codes)
            first = 1
        for k in range(first, len(self.levels)):
            l = self.levels[k]
            if self.mode == "reg":
                t = torch.ones(B, device=dev)
                tok = self._block_tokens(k, torch.zeros(B, l, STATE_DIM, device=dev),
                                         recon, plan, t)
                r_n = self._run(prefix, [tok])[0]
            else:
                r_n = torch.randn(B, l, STATE_DIM, device=dev) * temperature
                for i in range(nfe):
                    t0, t1 = i / nfe, (i + 1) / nfe
                    t = torch.full((B,), t0, device=dev)
                    pred = self._run(prefix, [self._block_tokens(k, r_n, recon, plan, t)])[0]
                    if self.mode == "diff":
                        a0, a1 = alpha_bar(torch.tensor(t0)), alpha_bar(torch.tensor(t1))
                        eps = (r_n - a0.sqrt() * pred) / (1 - a0).sqrt()
                        if eta > 0 and i < nfe - 1:     # DDPM-style ancestral step
                            sig = eta * ((1 - a1) / (1 - a0) * (1 - a0 / a1)).clamp_min(0).sqrt()
                            r_n = (a1.sqrt() * pred + (1 - a1 - sig ** 2).clamp_min(0).sqrt() * eps
                                   + sig * torch.randn_like(r_n))
                        else:
                            r_n = a1.sqrt() * pred + (1 - a1).sqrt() * eps
                    else:                                # Euler on the x1-prediction velocity
                        r_n = r_n + (pred - r_n) / (1 - t0) * (t1 - t0)
            r = r_n * self.lvl_std[k]
            levels_out.append(r)
            recon = recon + up(r, self.H) if self.residual else up(r, self.H)
        return (recon, levels_out) if return_levels else recon
