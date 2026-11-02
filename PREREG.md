# Pre-registration: confirmatory test (written 2026-10-30, before any confirmatory run)

## Exploratory finding (branch set v2, seeds 900000+, one training seed per model)

At fork anchors (true fall probability between 0.05 and 0.95, n = 280), the ratio of the
model's mean predicted fall probability to the true mean was:

| Model | Ratio | 95% bootstrap CI |
|---|---|---|
| CFM, 64 steps | 0.35 | [0.30, 0.41] |
| diffusion DDIM, 64 steps | 0.29 | [0.25, 0.34] |
| continuous NCM, levels 4..100, 64 steps | 0.75 | [0.66, 0.84] |
| continuous NCM, pyramid, 16 steps | 0.86 | [0.74, 0.98] |

Fork Brier score, NCM4@64 minus CFM@64: -0.0078 [-0.0147, -0.0019].

## Confirmatory design

- **Test set.** A fresh branch set: 320 episodes, seeds 950000 + 10i (test split; never
  generated or looked at before), 4 anchors x 128 branches x 100 steps, same generator
  (sim/generate2.py, unchanged).
- **Models.** Training seeds 0, 1, 2 for each of CFM, diffusion and NCM4 (levels
  4,8,16,32,64,100). Same data, architecture and budget (40k steps) as before. Seed 0 is
  the existing checkpoint.
- **Sampling.** 64 steps for all (deterministic samplers), 128 samples per anchor.
- **Fork set.** Anchors with true fall probability in (0.05, 0.95) on the fresh set.
- **Statistics.** Average the per-anchor quantities over the three seeds, then bootstrap
  over anchors (10,000 resamples) for 95% CIs.

## Hypotheses (all must hold to call the finding confirmed)

- **H1.** CFM and diffusion underestimate fork fall probability: the upper end of the
  ratio CI is below 0.6 for both.
- **H2.** NCM4 is less biased: its ratio is higher than CFM's, with a CI on the
  difference (NCM4 - CFM) excluding 0.
- **H3.** NCM4 has a lower fork Brier score than CFM, with the CI on the difference
  excluding 0.

## Controls (reported regardless of outcome)

- **CFM trained 2x longer (80k steps, seed 0).** If it closes most of the gap (ratio >
  0.6), the underestimation is partly undertraining and must be reported as such.
- **NCM4 without corruption (corrupt = 0).** Does the corruption training matter?
- **Full-trajectory accuracy** (energy distance) for every model, so calibration gains
  are not traded silently for worse samples.
