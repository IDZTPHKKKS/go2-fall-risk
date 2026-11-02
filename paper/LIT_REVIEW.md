# Literature review and reading guide

This file explains the papers the draft cites, grouped by topic: what each paper does, and
why it matters for our paper. Read the "must read" ones first. Every 2026 arXiv entry
below was checked on its arXiv page on 2026-10-31. The older papers are well-known
references.

## 0. Basics to learn first (in this order)

1. **Diffusion models.** Start with Lilian Weng's blog post "What are Diffusion Models?"
   (lilianweng.github.io, 2021). Then read DDPM (Ho et al. 2020) and DDIM (Song et al.
   2021). Learn three ideas: the noise schedule, x0-prediction, and deterministic DDIM
   sampling. Our diffusion baseline uses a cosine schedule (Nichol and Dhariwal 2021).
2. **Flow matching.** Read Lipman et al. 2023 ("Flow Matching for Generative Modeling") and
   rectified flow (Liu et al. 2023). Learn the idea of a straight path between noise and
   data, and Euler sampling. The "Flow Matching Guide and Code" (Lipman et al., 2024,
   arXiv 2412.06264) and the MIT course 6.S184 on flow matching and diffusion are good
   longer introductions.
3. **How diffusion and flow matching are related.** "Diffusion Meets Flow Matching: Two
   Sides of the Same Coin" (Gao et al. 2024, diffusionflow.github.io). This explains why
   CFM and diffusion behave alike in our results.
4. **Probability scoring.**
   - Brier score (Brier 1950): mean squared error between the predicted and true
     probability.
   - Proper scoring rules (Gneiting and Raftery 2007): why the Brier score rewards honest
     probabilities.
   - Calibration and isotonic regression (Guo et al. 2017; Zadrozny and Elkan 2002;
     Niculescu-Mizil and Caruana 2005).
   - AUC: how well a model ranks risky states above safe ones.
5. **Bootstrap confidence intervals** (Efron and Tibshirani 1994): resample the test anchors
   with replacement many times and look at the spread of the statistic.
6. **Pre-registration** (Nosek et al. 2018): write the hypotheses down before seeing the
   results. This is why our final test is believable.
7. **Transformers** (Vaswani et al. 2017): the network we use for every model.

## 1. Our method's parent: coarse-to-fine generation (must read)

- **Next-Concept Modeling (NCM)**: Kumbong et al., arXiv 2610.14162, Oct 2026.
  - Encodes text into levels of discrete residual codes and generates one level at a time,
    all codes in a level in parallel.
  - Uses block-diagonal attention with a prompt prefix, corruption of the coarser levels
    during training, and loss weights per level.
  - **C-NCM is this idea made continuous for trajectories.** We keep the levels, residuals,
    attention and corruption, and replace the VQ codes with flow matching at every level.
- **VAR**: Tian et al., NeurIPS 2024. Next-scale prediction for images, the best-known
  coarse-to-fine generator.
- **Cascaded diffusion**: Ho et al., JMLR 2022. Diffusion at growing resolutions.
- **Nested dropout**: Rippel et al., ICML 2014. Used in NCM's tokenizer to order
  information from coarse to fine.

## 2. Rare modes in diffusion and flow matching (must read: SubFlow)

These explain *why* flat models might underestimate falls.
- **SubFlow**: Lin et al., arXiv 2604.12273, Apr 2026.
  - Shows that flows trained with a squared loss learn a frequency-weighted *average* over
    sub-modes, so rare sub-modes get too little mass.
  - Their fix is to condition on a sub-mode label. This is the closest idea to our
    explanation of the bias.
- **Regional probability matching**: Huang et al., arXiv 2610.16780, Oct 2026. Flow
  matching gets mode frequencies wrong; they fix the mass of each region. Same problem,
  different setting (images and toy data).
- **Minority guidance** (Um et al., ICLR 2024) and **low-density sampling** (Sehwag et al.,
  CVPR 2022): ways to push diffusion toward rare samples.
- **Mode interpolation**: Aithal et al., NeurIPS 2024. Diffusion can create samples between
  modes. This was our first hypothesis (impossible "half-fall" futures). Our data did not
  support it, so the paper does not lean on it.
- **UK windstorms**: Tsoi et al., arXiv 2501.16110, 2025. A diffusion model underestimated
  the most intense storms. This is evidence from weather that diffusion misses extremes.

## 3. World models (know the main names)

- **Classic**:
  - World Models (Ha and Schmidhuber 2018).
  - Dreamer and DreamerV3 (Hafner et al. 2020, 2025).
  - DayDreamer, on real robots (Wu et al. 2022).
  - PETS, ensembles of probabilistic models (Chua et al. 2018).
- **Legged robots**: Robotic World Model (Li, Krause and Hutter, arXiv 2501.10100). A
  learned simulator for ANYmal and the Unitree G1. It is the closest robot setting to ours,
  but it does not measure failure probability.
- **Diffusion world models**:
  - Diffuser (Janner et al. 2022): plans by denoising trajectories.
  - DIAMOND (Alonso et al. 2024): Atari.
  - GameNGen (Valevski et al. 2025): DOOM.
  - DWM (Ding et al., arXiv 2402.03570): multi-step futures for offline RL.
  - PolyGRAD (Rigter et al., TMLR 2024).
  - Diffusion Forcing (Chen et al. 2024).
- **Main point for us:** all of these judge quality or return, not whether failure
  probabilities are right. That gap is our paper.

## 4. Evaluating world models and multi-future ground truth

- **Decision-centric evaluation**: Yu et al., arXiv 2606.15032, Jun 2026. A position paper:
  judge world models by the decisions they support, with calibration as one axis. **Cite
  it in the intro as motivation.**
- **dWorldEval**: Li et al., arXiv 2604.22152, Apr 2026. Uses a world model to estimate
  policy success rates.
- **Garden of Forking Paths**: Liang et al., CVPR 2020. Several possible futures for
  pedestrians: multi-future ground truth, but made by people, not by physics.
- **ForecastBench-Sim**: Lee et al., arXiv 2606.18686, Jun 2026. Re-seeds a strategy game
  (Freeciv) to score forecasts.
- **Twin Rollouts**: Ma et al., arXiv 2608.08982, Aug 2026. Forks a world model's noise
  for counterfactuals; it suggests forked simulator states but has no experiments yet.
- **Our difference:** exact copies of the physics plus controller state, with the
  probability of failure as the target.

## 5. Safety and failure prediction in robotics

- Learning to predict humanoid falls (Kalyanakrishnan and Goswami 2011).
- Safety critics (Srinivasan et al. 2020) and Recovery RL (Thananjeyan et al. 2021).
- **Latent disturbances**: Seo and Bajcsy, arXiv 2610.07599, Oct 2026. They model worst-case
  disturbances inside a world model for safe planning. It is close in spirit to our
  "pushes are the randomness" view.
- **FoMo-FD**: Huang et al., arXiv 2607.27511, Jul 2026. A flow-matching world model used to
  detect failures in surgical robots, calibrated on successful runs.
- **GenCast**: Price et al., Nature 2025. Diffusion weather ensembles, checked for
  calibration. A good example of how probabilistic forecasts *should* be evaluated.

## 6. Episode endings and censoring (for the data-pitfall section)

- Time Limits in RL (Pardo et al. 2018): how an episode ends changes what is learned.
- D4RL (Fu et al. 2020): a standard dataset where locomotion episodes end when the agent
  falls.
- Kaplan and Meier (1958): survival analysis for outcomes cut off before they are seen.
  Our pitfall is the same kind of selection effect.

## 7. Simulator

- MuJoCo (Todorov et al. 2012) and MuJoCo Menagerie for the Go2 model (Zakka et al. 2022).
- Raibert, *Legged Robots That Balance* (1986): the foot-placement rule our controller uses.

## Novelty check (to redo before submission)

We searched for papers that show diffusion or flow-matching robot world models
underestimating failure probability against branching ground truth, and did not find
one. We also did not find any paper on the fixed-window truncation pitfall.
Re-run these searches right before submitting:
- "world model failure probability calibration"
- "branching ground truth world model"
- "trajectory diffusion early termination windows"

New 2026 papers appear every week.
