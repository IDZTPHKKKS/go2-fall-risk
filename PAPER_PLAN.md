# Paper plan (draft, 2026-10-29)

## Thesis

Diffusion and flow-matching world models both turn a single Gaussian blob of noise into
futures through a smooth (Lipschitz) map. A smooth map cannot split one connected blob into
two separated outcomes (e.g. "the robot recovers" vs "the robot falls") without leaving
probability mass in between. For a physical system, "in between" means futures the
simulator never produces: half-falls, bodies hovering mid-tip, trajectories that average
two contact sequences. We expect three consequences at physical bifurcations:

1. Both families generate physically impossible futures ("mode interpolation").
2. The fall probability implied by their samples is miscalibrated, worst for rare
   failures.
3. More sampling steps or more capacity shrink the problem slowly, because the cause is
   the noise source, not the solver.

Fix: a **concept-first** world model.
- First pick a discrete "concept" for the future: a learned code for the coarse
  trajectory (residual VQ on the coarsest temporal level, as in Next-Concept Modeling),
  with an exact categorical distribution.
- Then fill in the continuous detail with a flow conditioned on that concept (coarse to
  fine residual levels).

Discrete choices can be separated with no mass in between. Event probabilities can be read
off exactly in one pass. Rare failures can be estimated by importance sampling, tilting
the categorical toward failure concepts and reweighting exactly.

## Math check: what the methods do, and what we may claim

**Diffusion (VP).**
- Forward noising: x_t = sqrt(ab_t) x + sqrt(1 - ab_t) eps, with eps ~ N(0, I).
- The network learns the denoiser E[x | x_t] (x0-, eps- and score-parameterizations
  are equivalent through Tweedie's formula).
- Sampling runs either the reverse SDE (stochastic, "ancestral") or the
  probability-flow ODE (deterministic, DDIM).
- In our code: cosine schedule, x0-prediction, DDIM (eta=0) and ancestral (eta=1).

**Flow matching (CFM, linear path, independent coupling).**
- Path: x_t = (1 - t) x0 + t x1, with x0 ~ N(0, I).
- The marginal velocity is u_t(x) = E[x1 - x0 | x_t = x] = (E[x1 | x_t = x] - x)/(1 - t).
- Sampling integrates this ODE. In our code: x1-prediction, Euler.

**They are one family.** Both are Gaussian paths x_t = alpha_t x1 + sigma_t eps:
- linear for CFM (alpha = t, sigma = 1 - t)
- trigonometric for cosine VP (alpha = sin(pi t/2), sigma = cos(pi t/2))

Both networks estimate the same quantity, E[x1 | x_t]. With exact networks, both
deterministic samplers transport N(0, I) exactly onto the data distribution. They differ
only in the loss weighting over noise levels, the schedule (and hence discretization
error) and the sampler (Kingma & Gao 2023; Gao et al. 2024, "Diffusion meets flow
matching"). This is why diffusion-vs-CFM comparisons on the same backbone tend to tie, as
in our course project. It also means a failure shared by the family is the more
interesting finding. Minibatch-OT coupling is genuinely different (it changes the
path), so it is a separate baseline.

**What we must NOT claim.** We must not say "diffusion/CFM cannot represent bimodal
futures". With the exact velocity or score they represent any distribution,
bifurcations included.
- For well-separated outcomes, the ideal flow map is extremely steep (nearly
  discontinuous). Its Lipschitz constant grows with mode separation over within-mode
  spread.
- Learned networks are smooth: finite data, finite capacity, the bias of neural nets
  toward smooth functions, and few-step solvers.
- A push-forward of a Gaussian through an L-Lipschitz map has connected support. Its
  mass between two separated sets is bounded below in terms of L and the gap (Salmona et
  al. 2022, NeurIPS).
- For stochastic samplers, the connectedness argument does not apply directly. But a
  smooth learned score still interpolates across the low-density gap, and diffusion
  models empirically produce "mode interpolation" samples (Aithal et al. 2024, NeurIPS).

**So our claim is about learned, finite-step models, and it is empirical plus mechanism:**
- (i) how much mass lands between physical outcomes, for both samplers
- (ii) how it scales with NFE and capacity
- (iii) what it does to fall-risk calibration

**1-step sampling collapses.** With x1-prediction, the first step sees pure noise, which
is independent of the target, so an ideal network outputs E[x1 | context]: the mean
future, whatever the noise is. So one step means no diversity. NCM is diverse in one pass
per level only because its levels are categorical. Continuous levels need several steps.
This favors a discrete first level.

**NCM (checked against the paper, arXiv 2610.14162).**
- Multiscale residual VQ tokenizer: the residual is pooled down to l_k vectors,
  quantized by nearest code, reconstructed up and accumulated, with nested dropout.
- The generator predicts all codes of a level in parallel. Codes within a level are
  sampled independently given the coarser levels. That factorization is why coarse
  levels must carry the correlations.
- Attention: block-diagonal plus a prompt prefix. The prompt cannot see the targets.
- Loss weights: w_k ∝ l_k^-alpha.
- Corruption: conditioning codes are replaced by uniform random codes of the same
  level, at a rate decreasing linearly across levels with mean 0.5, to counter the
  train-test mismatch.
- Our continuous version: per-level N(0, 1) replacement at rates from 1.5c down to 0.5c
  (c = 0.5) over the conditioning levels. The concept level uses a separate small rate
  (0.1), because a concept sampled from a calibrated categorical is not an "error".
  (Fixed 2026-10-29: the first draft used rate 1.0 on the coarsest level, which
  destroyed it as conditioning.)

**Concept-first as a mixture.** p(x | c) = sum_k p(k | c) p(x | k, c).
- A discrete choice is a free discontinuity: components separated by the concept need
  no mass in between, even with smooth conditionals.
- Caveat: k-means concepts need not align with the fall / no-fall boundary. If a
  concept straddles it, its flow must still split. So we report concept purity (how
  close P(fall | k) is to 0 or 1).
- P(fall) = sum_k p(k | c) P(fall | k) is closed-form (no sampling noise). But P(fall | k)
  is estimated from training data, so we call it "closed-form", not "exact".

**Ground-truth caveats (state them in the paper).**
- Branches condition on the full simulator state. The model sees 0.5 s of state
  history, the commands, the physics parameters and a 4 m terrain map that covers
  everything reachable within the horizon. It does not see controller internals
  (integrators, foot liftoff points). So branch spread is a slight underestimate of the
  Bayes predictive for the model's inputs.
- Pushes are memoryless (Poisson), so the past adds no information about future pushes.

## What is new (and what is not)

Known already:
- Push-forward models with Lipschitz maps struggle with multimodal targets (Salmona et
  al. 2022).
- Diffusion "hallucinations" come from mode interpolation (Aithal et al. 2024).
- Hybrid discrete-plus-continuous image generators exist (HART, 2024).

New here:
- **Branching ground truth for world models.** For each anchor state, re-simulate many
  futures from the identical simulator state, varying only the exogenous randomness
  (future pushes). This gives the *true conditional distribution*, so we can measure
  calibration, coverage and impossible-sample mass directly, instead of scoring against
  the single future that happened. It is a public dataset contribution.
- **Showing that diffusion and CFM world models produce impossible futures and
  miscalibrated risk at physical bifurcations**, how this scales with NFE and capacity,
  and that it is shared by both families (it explains why CFM vs diffusion comparisons
  tie).
- **A concept-first hybrid world model with exact event probabilities**, plus importance
  sampling of rare failures with the model as the proposal.

## Other directions considered (kept as ablations or backups)

1. **Continuous NCM.** Residual temporal pyramid, one flow pass per level, corruption of
   the coarser levels. It is close to HiFlow (policies) and FlowAR (images), so on its own
   it reads as "X applied to robots". Here it is the continuous half of the concept-first
   model and an ablation (all levels continuous vs a discrete first level).
2. **Mixture-source flow matching.** Replace the single Gaussian noise source with a
   mixture whose component is chosen by a classifier. It is mathematically close to
   concept-first, and a good unifying view for the theory section.
3. **Branch-supervised training.** Train diffusion and CFM on several simulated futures
   per context, instead of one. This control experiment asks whether better data alone
   fixes mode interpolation. If not, the cause is the model class, which strengthens the
   thesis.
4. **Risk-tilted sampling.** Importance sampling for rare failures using the exact concept
   probabilities, for sample-efficient fall-probability estimates (safety evaluation).
5. **Physics-consistency score.** Inverse-dynamics feasibility of generated futures
   (torques within limits, no ground penetration), as an impossible-future metric that
   does not need branches. It also works on real-robot data, where branches are
   impossible.

## Next ideas (decide after the first branch-eval results)

Within the Go2 study:
- **Contact-mode concepts.** Legged robots are hybrid systems. Make the discrete concept
  the foot-contact schedule instead of k-means clusters, then a flow for the motion.
  Metrics: foot penetration, sliding during contact, contact-timing error.
- **Closed-loop use.** Pick commands by the model's one-pass fall probability (risk-aware
  command selection), then count falls in sim with vs without it.
- **Rare-event estimation** by importance sampling over concepts: samples needed to
  estimate the fall rate within +/-10%.
- **Decision time.** When do branches separate, and do models' uncertainties rise at
  that time?
- **Branch-supervised training** of diffusion/CFM (all 128 futures per state) as a
  control experiment.

More domains (same thesis, branching ground truth, MuJoCo on KIAC CPUs):
- **Planar pushing / toppling.** A finger pushes a box, which slides, spins left or right,
  or tips over.
- **Block-tower collapse.** Many discrete outcomes (direction, partial collapse).
- **Policies.** Diffusion/flow policies on demos with two ways around an obstacle (plus
  Push-T): do they blend modes and collide at the decision point? Compare with a
  concept-first policy. Related work to position against: BeT, VQ-BeT (discrete action
  codes plus offsets).

## Data (MuJoCo Go2, generated on KIAC CPUs)

Generator v2 changes:
- **Memoryless pushes.** Pushes follow a Poisson process instead of fixed intervals, so
  the future push distribution does not depend on hidden history. This is what makes
  branching ground truth equal to the Bayes predictive the models should learn.
- **Known disturbance parameters.** Push level, friction, payload, motor strength and
  damping are logged and given to the models as conditioning.
- **Short state history** (0.5 s) in the conditioning.
- **Branching mode.** Copy MjData and the controller, then roll out N futures with fresh
  push draws, holding the commands fixed.
- **Controller fix** for fast gait switches. This is the cause of the 8% of falls that
  happen with no push.

## Models (one shared transformer backbone, same budget)

- regression
- diffusion (x0-prediction)
- CFM (x1-prediction, independent and OT coupling)
- C-NCM: continuous residual levels with corruption
- concept-first hybrid

## Metrics

All are computed against the branching ground truth:
- **Impossible-sample mass:** distance to the nearest true branch, and a physics
  plausibility check.
- **Fall-probability calibration:** Brier score, ECE, and log-loss on the per-anchor
  fall fraction.
- **Distributional distance:** energy distance and MMD per anchor.
- **Rare-event estimation error** versus number of samples.
- **NFE and wall-clock time.**

## Pre-registration

Freeze the hypotheses, metrics and test split before any test-set evaluation (we learned
this from the course project).

## Related work found in the 2026-10-30 literature search (must cite and position against)

- **SubFlow (arXiv 2604.12273).** Closest to our finding. Class-conditional flow matching
  trained with MSE learns a frequency-weighted average of sub-modes, so it
  over-represents dominant modes and neglects rare ones, and more sampling steps do not
  fix it. Their fix is conditioning on a sub-mode index (close to our shelved
  concept-first idea). Our contribution relative to it: shown in robot world models,
  where the rare mode is failure, measured against exact branching ground truth, with
  consequences for risk calibration, and fixed by coarse-to-fine residual generation
  rather than sub-mode labels.
- **arXiv 2610.20035.** Flow-matching trajectories settle coarse features first, then
  fine details. Relevant to the mechanism (why a coarse level may isolate the
  fall / no-fall decision).
- **Branching ground truth.**
  - ForecastBench-Sim (2606.18686) proposes N re-seeded rollouts per start state, but only
    as future work.
  - Twin Rollouts (2608.08982) forks the simulator for counterfactual video targets.
  - BranchDrive (2609.27275) is a branch-structured driving dataset.
  - CivWorldBench (2610.16450) uses re-seeded futures.
  - Garden of Forking Paths (CVPR 2020) gives multi-future pedestrian ground truth.
  - Ours: exact physics-state forks for robot dynamics, with failure-probability ground
    truth.
- **Diffusion for failure sampling** (DiFS, 2406.14761; DiffTilt, 2607.23134). Diffusion
  is used as a proposal for finding rare failures, which is a different goal.
- **Coarse-to-fine neighbours.**
  - WAM-Flow (CVPR 2026) is discrete flow over denoising steps, for driving.
  - Weather super-resolution residual flows (2604.00897).
  - VAR, FlowAR (images) and HiFlow (policies).
  - None of these is a continuous scale-wise residual world model with corruption.
- **No paper found yet** that shows diffusion/FM robot world models underestimating
  failure probability. Re-check before submission, and read SubFlow in full.
