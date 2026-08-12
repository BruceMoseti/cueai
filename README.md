# CueAI

[![CI](https://github.com/BruceMoseti/cueai/actions/workflows/ci.yml/badge.svg)](https://github.com/BruceMoseti/cueai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A physics simulator for billiards, a closed-form solution that replaces it, and
a learned model that corrects what the closed form misses — roughly 60,000x
faster than integration, with the accuracy measured and the failure mode stated.**

Predicting where the balls come to rest costs **about 37 seconds per rack shot**
by numerical integration. This project reduces that to **0.58 ms** with a mean
error of **97 mm for direct shots** on a 2.54 x 1.27 m table, and — the part that
makes it usable — tells you in advance which of its own predictions to trust.

Every number below comes from this repository: `make all` regenerates the tables
and figures from scratch, and `make test` checks the physics claims against their
closed-form references.

![Draw, stun and follow from the same stroke speed](docs/assets/spin_control.png)

*One stroke speed, three cue tip heights. Backspin brings the cue ball back
behind where it started, a centre-ball hit stops it dead at the object ball,
topspin sends it through. All three come out of the cloth model, not from
special-casing.*

---

## Results

Held-out test set of 4,000 shots from 20,000 simulated shots. Error is the
distance between the predicted and the simulated resting position, averaged over
the cue ball and the object ball. Full tables in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Method | Cost per shot | Mean error | Direct shots (no cushion) | R² |
|---|---:|---:|---:|---:|
| Numerical simulator, 16 balls | 36.8 s | — (ground truth) | — | — |
| Closed-form solver, no fitting | 0.26 ms | 494 mm | 114 mm | 0.01 |
| Gradient boosting on the same features | 2.5 ms | 385 mm | 160 mm | 0.51 |
| **Closed form + learned residual** | **0.58 ms** | **378 mm** | **97 mm** | 0.42 |

The learned residual has the lowest error of the three, is four times cheaper to
evaluate than the boosted trees, and is the only one that improves on the physics
for the shots where physics is nearly sufficient. Gradient boosting posts the
higher R² by hedging toward the middle of the table on shots nobody can predict,
which flatters the variance-explained metric and costs it 60 mm on the shots that
matter.

### What that average hides

Two thirds of sampled shots never actually reach the object ball. It stays where
it started, which the closed-form baseline predicts *exactly*, so those rows
donate a free zero to half of the error metric. Splitting them out, in mm:

| Ball-ball contact | Share | Closed form cue / object | Boosting cue / object | CueNet cue / object |
|---|---:|---:|---:|---:|
| no | 67.6% | 585 / **0** | 468 / 76 | 411 / 18 |
| yes | 32.4% | 1032 / 797 | **638 / 606** | 738 / 701 |

Two things fall out of this that are worth saying plainly.

The residual formulation earns its keep on the shots where nothing happens: it
adds 18 mm of spurious object-ball motion against gradient boosting's 76 mm,
because "predict zero correction" is its default rather than something it has to
learn.

And it loses on the shots where a collision has to be modelled: 738 mm against
638 mm for the cue ball. That is the cost of anchoring to a baseline with a blind
spot — the closed-form solver has no ball-ball contact model at all, so on a third
of the shot space the residual is asked to undo a metre of error rather than
refine a good guess. Giving the closed form even a crude ghost-ball collision
would be the highest-value next change, and it is a physics change, not an ML one.

### Where it stops working, and why that is the interesting part

![Prediction error by cushion contacts](docs/assets/accuracy.png)

A resting position is a smooth function of the shot until the ball starts
ricocheting between cushions. After two or three rail contacts, a millimetre of
cue placement moves the outcome by a table length. The error breakdown above is
published instead of hidden inside a single average, because "this model predicts
direct and one-rail shots to about 10 cm and multi-rail scatter not at all" is a
usable statement, while "378 mm mean error" is not.

### Knowing which predictions to trust, before making them

![Error against coverage](docs/assets/reliability.png)

The breakdown above is sliced by what the simulator *did*, which you only know
after paying the 37 seconds. That makes it an autopsy, not a control.

But the closed-form solver reports how many cushions it *expects* on the way to
its answer, and that number is already computed as part of the prediction, so it
is free. It turns out to be a good enough proxy for "is this shot chaotic" to use
as a gate:

| Answer only when the solver expects | Coverage | Mean error |
|---|---:|---:|
| No cushion | 9.8% | **100 mm** |
| At most one cushion | 27.5% | 189 mm |
| At most two cushions | 50.0% | 253 mm |
| Anything (no gate) | 100% | 378 mm |

So the fast path is not a 378 mm model. It is a 100 mm model that knows it should
decline nine shots in ten, or a 253 mm model over half the shot space, and the
shots it declines can be sent to the simulator. A surrogate that reports its own
applicability domain can be deployed; one with a single headline error cannot.
The learned residual is also the most accurate of the three models at every
coverage level, which is a stronger claim than its 1.9% edge on the overall mean.

### Speed

![Cost of one shot prediction](docs/assets/latency.png)

## The physics is verified, not asserted

![Simulator versus closed form](docs/assets/validation.png)

The simulator is tested against closed-form solutions and conservation laws, not
against its own earlier output. A struck ball must begin rolling at exactly
`5/7` of its launch speed after sliding `12v₀²/(49 μ_s g)`; frictional collisions
must conserve momentum to machine precision; no contact may create energy. See
[docs/VALIDATION.md](docs/VALIDATION.md) for the full table of properties,
tolerances and measured deviations — and for the list of effects deliberately not
modelled, such as cue ball squirt and swerve.

This mattered. The original implementation had the contact-point slip velocity
and the friction torque in opposite handedness, so friction drove a struck ball
*away* from rolling: it slid until it stopped, the rolling speed was `0` instead
of `5/7 v₀`, and stopping distances were four times short. Nothing in the code
looked wrong. A closed-form comparison found it immediately.

![16-ball break](docs/assets/break_shot.png)

## Try it

```bash
git clone https://github.com/BruceMoseti/cueai && cd cueai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make test          # 44 tests, including the closed-form validation suite
make all           # regenerate the dataset, model, benchmarks and figures
```

`make all` takes about 20 minutes end to end on eight cores, of which 15 are
simulating the 20,000 training shots. Generation is seeded per sample, so the
dataset it produces is identical to the one behind the numbers above.

### Serve predictions

```bash
make api           # http://localhost:8000/docs
```

```bash
# Sub-millisecond estimate: closed form plus learned residual
curl -s localhost:8000/predict/fast -H 'content-type: application/json' \
  -d '{"speed": 2.2, "angle_deg": 8, "english_y": -0.3, "cue_x": 0.6, "cue_y": 0.6}'

# Full simulation, with the fast estimate alongside and the gap between them
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"speed": 2.2, "angle_deg": 8, "full_rack": false, "obj_x": 1.4, "obj_y": 0.7}'
```

### Interactive table

```bash
pip install -e ".[ui]"
make ui
```

Drag balls, aim, shoot. Each rack shot runs the reference simulator, so expect a
few seconds of thinking time — which is the entire motivation for the fast path.

## How it works

Three tiers, each with a different accuracy and cost, described in full in
[docs/DESIGN.md](docs/DESIGN.md).

**1. Numerical simulator.** Four-state cloth dynamics (sliding, rolling,
spinning, stationary) integrated at 1 ms, frictional ball-ball impulses with spin
transfer and throw, cushion rebound with speed-dependent restitution, pocket
capture, and a spatially varying cloth friction field.

**2. Closed-form solver.** No integration. While a ball slides, its slip velocity
decays along a *fixed* direction, so the friction force is constant and the path
is exactly a parabola of known duration. Rolling is a straight line. A shot
becomes a handful of exactly solvable segments joined at cushions, where the
normal velocity is damped while the spin term carries through. That last detail
is what a plain mirror-reflection approximation gets wrong: it disagreed with the
simulator by about a metre, where this solver lands within 114 mm on direct shots
and 225 mm across one rail.

**3. Learned residual.** A small MLP predicts the vector from the closed-form
endpoint to the simulated one. Its output head starts at zero, so training begins
from "trust the physics" and departs only where the data insists. Its inputs
include the closed-form solver's own conclusions — predicted endpoint, expected
cushion count, expected pot, ghost-ball contact geometry. An ablation on the same
architecture, epochs, seed and split puts a number on that: fed raw shot
parameters alone it scores 173 mm on direct shots, worse than the 114 mm of the
physics it was meant to improve; fed the solver's conclusions as well, 97 mm.

## What this demonstrates

Billiards is the domain; the transferable content is below.

**Surrogate modelling of an expensive simulator.** The pattern — establish a
trusted reference, find the closed-form structure inside it, learn only the
residual, and quantify the accuracy you traded for the speed — is the same one
used for pricing engines that are too slow for intraday risk, finite-element
models too slow for design loops, and any Monte Carlo where the inner loop is the
bottleneck. The four orders of magnitude here come mostly from the closed form,
not from the network, which is usually how it goes.

**Validating against theory rather than against yourself.** A snapshot test
would have locked in a sign error that made the central physics wrong. Closed-form
references, conservation laws and analytic decay rates caught it in one run. The
same discipline applies to any numerical pipeline: solve a special case exactly
and check against it.

**Selective prediction with a free confidence signal.** The per-stratum error
breakdown, the prediction-spread ratio and the coverage curve exist so a caller
knows which predictions to trust — and the gating signal falls out of the physics
baseline at no extra cost, rather than requiring a second calibrated model. A
surrogate that is confident everywhere and accurate in half the space is more
dangerous than the slow model it replaced.

**Model selection without leaking the test set.** Epochs are chosen on a
validation split carved from the training data; the test split is scored once,
after training. The GBM comparison uses the identical split.

**Reproducibility as a property of the code.** Dataset generation is parallel and
seeded per sample, so the data is byte-identical on 1 core or 32. Training and
serving construct features through one function, and a test pins the two paths
together, because train/serve skew degrades a model quietly instead of failing
loudly.

**Deployability.** The residual model exports to ONNX and runs through ONNX
Runtime with no PyTorch in the serving path. PyQt and OpenCV are optional extras,
so the package installs headless. CI lints, type-checks and tests on three Python
versions, then runs the whole data to model to benchmark to figures pipeline and
uploads the artefacts.

## Repository map

```
src/cueai/
  physics/
    ball.py         four-state cloth dynamics for one ball
    collisions.py   frictional ball-ball impulses, cushions, pockets
    simulator.py    the 16-ball reference simulator
    analytic.py     the closed-form solver and its derivations
    rack.py         8-ball rack geometry and ball identities
  ml/
    dataset.py      parallel, per-sample-seeded shot generation
    features.py     one feature path for training and serving
    model.py        CueNet, zero-initialised residual head
    train.py        training, baseline comparison, stratified evaluation
    infer.py        predict_fast (0.58 ms) and predict (full simulation)
  api/main.py       FastAPI service
  ui/app.py         PyQt6 interactive table
  vision/overlay.py OpenCV trajectory overlays

tests/
  test_validation.py  closed-form physics validation
  test_features.py    train/serve feature consistency
  test_physics.py     simulator behaviour
  test_metrics.py     the reported metrics, including the trust gate
  test_api.py         HTTP contract

scripts/
  benchmark.py      writes docs/BENCHMARKS.md
  make_figures.py   writes docs/assets/*.png
```

## Limitations

- Not calibrated against a real table. No measurements were taken, so the claim
  is internal consistency with classical mechanics, not fidelity to specific
  equipment. Cloth and cushion coefficients come from the published ranges.
- Cue ball squirt and swerve are not modelled, so aiming advice would be
  systematically off for heavy sidespin.
- Multi-rail outcomes are not usefully predictable by any of the three methods
  here, as measured above.
- The fast path predicts *where balls stop*; it is not an aiming aid. On the layout
  in [docs/DESIGN.md](docs/DESIGN.md) the window that pots a ball is a quarter of a
  degree wide — `tests/test_validation.py` asserts that the ghost-ball line pots and
  half a degree off misses — roughly ten times finer than the surrogate resolves.
  Aiming is exact closed-form geometry anyway, and needs no model.
- The reference simulator is straightforward Python. It is a definition of truth,
  not a fast engine; optimisation stopped once the collision loop was no longer
  the obvious bottleneck.

## References

The cloth and collision model follows the standard treatment in Ron Shepard's
*Amateur Physics for the Amateur Pool Player*, Wayland Marlow's *The Physics of
Pocket Billiards*, and David Alciatore's technical proofs; the four-state
formulation matches the approach taken by
[pooltool](https://github.com/ekiefl/pooltool). Coefficients are from the
published ranges in those sources.

## License

MIT — see [LICENSE](LICENSE).
