# Will It Fall? Fall risk in generative world models (Unitree Go2)

Do diffusion and flow-matching world models predict robot failures as often as they really
happen? We simulate a Unitree Go2 in MuJoCo walking on varied terrain while random pushes
hit it. At each test state we fork the simulator (physics and controller state) 128 times,
so we know the true probability that the robot falls in the next 2 s. Then we ask each world
model for 128 sampled futures from the same state and count how many fall.

Draft paper: [`paper/main.pdf`](paper/main.pdf).

## Main results (test set: 1,149 states, 319 forks with 0.05 < p < 0.95)

| Model | Predicted / true fall risk at forks | Fork Brier | AUC |
|---|---|---|---|
| CFM (64 steps) | 0.65 [0.57, 0.74] | 0.0246 | 0.899 |
| Diffusion (64 steps) | 0.57 [0.50, 0.65] | 0.0254 | 0.894 |
| C-NCM, coarse-to-fine (64 steps per level) | **1.05** [0.95, 1.17] | 0.0274 | 0.902 |
| Classifier, same inputs and network | 0.92 [0.82, 1.04] | 0.0271 | 0.881 |

- Logging stopped 1 s after a fall in our first dataset, which hid early falls from the
  fixed-length training windows and halved the predicted risk (CFM 0.36). All numbers
  above use the fixed data.
- More sampling steps help the flat models slowly: CFM reaches 0.83 and diffusion 0.77 at
  512 steps. C-NCM reaches 0.83 with 96 network passes and 0.97 with 192.
- C-NCM does not rank states better: the Brier difference to CFM is not significant, and
  isotonic recalibration of CFM on held-out data matches it.
- The final test was pre-registered (`PREREG_v3.md`): H1 and H2 held, H3 and H4 did not.

## Layout

```
sim/            simulator, controller, data generation
  go2_ctrl.py     model-based Go2 controller (50 Hz gait/balance, 500 Hz joint PD)
  terrain.py      flat / rough / slope / stairs heightfields
  generate2.py    episodes with Poisson pushes; --anchors forks 128 branches per anchor
models/         world models, training, evaluation, statistics
  c2f.py          one model class: CFM, diffusion, C-NCM (levels), concept-first
  data.py         training windows and branch sets (refuses short post-fall tails)
  train.py, evaluate.py, v3_stats.py, fall_classifier.py
  *.sbatch        the exact cluster jobs behind each result
paper/          LaTeX draft, figures, figure scripts, reading guide (LIT_REVIEW.md)
results/        small local test outputs (main result files: paper/data/)
PREREG.md       pre-registration of the first test (on the truncated data)
PREREG_v3.md    pre-registration of the final test
```

## Setup

Python 3 with MuJoCo 3.x, PyTorch 2.x, NumPy, h5py and matplotlib.

The Go2 model comes from MuJoCo Menagerie. Check it out at the commit we used and add our
terrain scene:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie assets/menagerie
cd assets/menagerie
git sparse-checkout set unitree_go2
git checkout f3b6013629c7b94f8ee93fd977b161fa8350f9bc
cp ../patches/unitree_go2/scene_terrain.xml unitree_go2/
cd ../..
```

## Reproducing

The cluster jobs in `sim/*.sbatch` and `models/*.sbatch` run these steps in parallel.

```bash
cd sim
# training episodes (one of 64 shards; seeds 0-6015 in total)
python generate2.py --start 0 --count 94 --post-fall 100 --out ../data/v3/train_000.h5
# test branch set (one of 64 shards; seeds 970000 + 10i)
python generate2.py --start 970000 --count 5 --stride 10 --anchors 4 --branches 128 \
    --horizon 100 --post-fall 100 --out ../data/final/branch_000.h5

cd ../models
python data.py ../data/v2/terrain_cache.h5 "../data/v3/train_*.h5" "../data/final/branch_*.h5"
D='--data ../data/v3/train_*.h5 --horizon 100 --batch 256 --duty 1 --d 256 --depth 4 --steps 40000'
python train.py --name v3_cfm_s0  $D --seed 0 --levels H
python train.py --name v3_diff_s0 $D --seed 0 --levels H --mode diff
python train.py --name v3_ncm4_s0 $D --seed 0 --levels 4,8,16,32,64,100
python evaluate.py --runs v3_cfm_s0 --nfe 64 --branches "../data/final/branch_*.h5" --tag _v3_final_cfm
python v3_stats.py                 # pre-registered statistics -> results/v3_stats.json

cd ../paper
python scripts/make_figs.py        # figures and the main table
tectonic -X compile main.tex
```
