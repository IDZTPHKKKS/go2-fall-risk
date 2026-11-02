# ncm-robotics: working notes

Goal: a continuous Next-Concept-Modeling (NCM, arXiv 2610.14162) world model for the
Unitree Go2, compared against CFM / diffusion / regression, plus a public MuJoCo dataset.
Runs only on this Mac. Use `taskpolicy -b` (efficiency cores) for long jobs so the Mac
stays cool.

## Status (2026-10-27, evening)
- `sim/go2_ctrl.py`: controller, now terrain-aware.
  - Swing targets use the terrain height.
  - Body height is held relative to the stance feet.
  - Swing feet rise early when stepping up.
  - Fixed a roll/pitch frame bug that made the robot fall whenever it wasn't facing +x.
- Gaits: stand, walk, trot, bound, and pace (period 0.3 s, duty 0.6).
- `sim/terrain.py`: one 16 x 16 m heightfield (0.04 m cells) with flat / rough / slope /
  stairs, all with a flat start pad. Checked against MuJoCo ray casts.
- Terrain test (0.5 m/s, 4 headings): flat, rough and slope 4/4 for every gait.
  Stairs: trot 4/4, bound 4/4, pace 3/4, walk 2/4.
- `sim/generate.py`: episode generator.
  - 10 s at 50 Hz; after a fall it keeps logging 1 s, then stops.
  - Randomized: foot friction 0.4-1.2, payload 0-3 kg, motor strength +/-10%, joint
    damping x0.7-1.3, heading.
  - Commands are segments of 1.5-4 s.
  - Push level per episode: none / light / medium / strong.
  - Logs per step: qpos, qvel, 46-d state, LowCmd (q_des, dq_des, kp, kd, tau_ff), applied
    tau, cmd, gait, phase, scheduled and actual contacts, contact forces, feet, push,
    7x7 height scan, ground z.
  - About 13 s and about 260 KB per episode on the efficiency cores.
- Open issue: the first test had a high fall rate (4/6). Diagnosing whether falls come from
  pushes (fine) or the controller (fix before generating thousands).

## Method plan (continuous NCM)
- Residual temporal pyramid: area-pool down, linear up.
- One shared network predicts each level's residual in a single pass.
- Noise corruption of the coarse conditioning, linearly decreasing over levels.
- Block-diagonal attention with a prefix.
- Ablations: residual vs full levels (HiFlow-style), corruption on/off, 1 level vs many,
  learned-AE levels.

## Related work to position against
- HiFlow arXiv 2603.27281
- CARP (ICCV 2025)
- WALL-SS arXiv 2608.26239
- FlowAR
