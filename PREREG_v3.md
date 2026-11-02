# Pre-registration v3: rerun on corrected training data (written 2026-10-31, before any v3 result)

## Why a rerun

The v2 training data stopped logging 50 steps after a fall, while every training window
needs 100 future steps. So no training window ever showed a fall in its first 51 steps
(49% of true branch falls happen there): a survivorship bias in the learned fall
probability. The v2 confirmatory test (PREREG.md) passed H1-H3, but its effect is
confounded by this bias, and an isotonic recalibration of CFM erased NCM4's Brier
advantage. The generator now logs 100 steps after a fall (schema 2.1,
sim/generate2.py --post-fall), and models/data.py refuses data with a shorter tail.

## Design

- **Training data.** data/v3: the same 6,016 episodes (seeds 0-6015) regenerated with
  100 post-fall steps (sim/gen_v3.sbatch, job 72371; models/verify_v3.py checks it).
- **Final test set.** data/final: 320 fresh episodes, seeds 970000 + 10i, never generated
  or looked at before, 4 anchors x 128 branches x 100 steps, same generator.
- **Calibration set.** data/confirm (the v2 confirm set, 1,163 anchors), used only to fit
  the recalibration maps.
- **Models.** v3_cfm, v3_diff, v3_ncm4 (levels 4,8,16,32,64,100), seeds 0, 1, 2; settings
  identical to the v2 runs (40k steps, d 256, depth 4, batch 256). Jobs 72372, 72373.
- **Sampling.** 64 steps, 128 samples per anchor.
- **Fork set.** Final-set anchors with true fall probability in (0.05, 0.95).
- **Statistics.** Per-anchor predicted probability averaged over the 3 seeds; bootstrap
  over anchors (10,000 resamples), 95% CIs; paired differences on the same anchors.
- **Recalibration baseline.** For each model family, an isotonic (PAV) map from the
  seed-averaged predicted probability to the true probability, fit on all calibration-set
  anchors and applied unchanged to the final set.

## Hypotheses

- **H1.** CFM and diffusion still underestimate fork fall probability on clean data:
  upper end of the ratio CI (mean predicted / mean true) below 0.8 for both.
- **H2.** NCM4 is less biased than CFM: the CI on the ratio difference (NCM4 - CFM)
  is above 0.
- **H3.** NCM4 has a lower fork Brier score than CFM (raw): CI on the difference below 0.
- **H4.** NCM4 (raw) has a lower fork Brier score than recalibrated CFM: CI below 0.

Interpretation fixed in advance: if H1 fails, the v2 underestimate was the data bug and
is reported as such; a claim that NCM4 is better then needs H3 and H4.

## Reported regardless of outcome

All-anchor Brier and cross-entropy; discrimination (AUC); energy distance; calibration
by bins of true probability; recalibrated NCM4 vs recalibrated CFM; the v2 seed-0
models (cfm, diff, ncm4) on the final set, which measures the size of the data bias.
