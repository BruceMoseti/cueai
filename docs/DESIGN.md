# Design

## The problem this solves

Simulating a billiards shot accurately is slow. Sixteen bodies, each in one of
four motion states, with collisions resolved every millisecond, costs about
**37 seconds per rack shot** in this implementation. Anything that needs many
shot outcomes — searching for a good shot, estimating the odds of a break,
training a policy — cannot afford that.

The usual answer is to replace the simulator with a learned model. Done naively
that throws away physics that is already exact, and it produces a model whose
errors have no structure. The approach here keeps the physics and learns only
what the physics leaves out.

```
shot parameters (speed, angle, tip offset, cue position, cloth μ, cushion e)
        |
        +-----------------------------+
        |                             |
        v                             v
  closed-form solver             CueNet residual
  (exact segments, 0.26 ms)      (what the closed form misses)
        |                             |
        +--------------+--------------+
                       v
              predicted resting positions        0.58 ms total
                       :
                       : compared against
                       v
              numerical simulator                37 s, ground truth
```

## Three tiers, on purpose

**Tier 1: the numerical simulator** (`cueai.physics.simulator`). Explicit Euler
integration at 1 ms, four-state cloth dynamics, frictional ball-ball impulses
with spin transfer, cushion rebound, pocket capture. This is the definition of
truth for everything else, and it is validated against closed-form results in
[VALIDATION.md](VALIDATION.md).

**Tier 2: the closed-form solver** (`cueai.physics.analytic`). No integration at
all. The observation that makes this possible: while a ball slides, the slip
velocity `u` decays along a *fixed direction*, so the friction force is constant
and the path over that phase is exactly a parabola of known duration
`|u|/(3.5 μ_s g)`. Rolling is a straight line of length `v²/(2 μ_r g)`. A shot is
therefore a short sequence of exactly solvable segments joined at cushion
contacts, where the normal velocity is damped and the spin term `u - v` carries
through — which is why a ball leaves a rail sliding rather than rolling. Getting
that detail right matters: a plain mirror-reflection approximation, which assumes
the ball leaves the rail rolling, disagreed with the simulator by about a metre,
while this solver lands within 114 mm on direct shots and 225 mm across one rail.

**Tier 3: the learned residual** (`cueai.ml`). A small MLP predicts the vector
from the closed-form endpoint to the simulated endpoint. Its head is initialised
to zero, so training starts from "trust the physics exactly" and moves away only
where the data insists.

## Why the features matter more than the architecture

The first working version of this model made predictions *worse* than the
closed-form solver on the easiest shots. The network was being fed raw shot
parameters, so to know where the closed-form model went wrong it first had to
rediscover cushion reflection geometry from a speed and an angle. It spent its
capacity on the chaotic shots that dominate the loss and degraded the clean ones.

The fix was to hand the network what the physics already knew: the closed-form
endpoint, the cushion count it expects, whether it expects a pot, and the
ghost-ball geometry deciding whether the object ball is contacted at all.

Every training run re-measures this as an ablation, on the same architecture,
epochs, seed and split, so the claim is not a story about an earlier commit:

| CueNet inputs | All shots | Direct shots |
|---|---:|---:|
| Raw shot parameters only | 473 mm | 173 mm |
| Plus the closed-form solver's output | 378 mm | 97 mm |
| _closed form alone, for reference_ | 494 mm | 114 mm |

Without the physics features the network is *worse than the physics* on the shots
the physics nearly solves — 173 mm against 114 mm — while still beating it on
average, which is exactly the failure that a single headline number conceals. With
them it is better on both. The predicted cushion count does double duty: it also
lets the model recognise a chaotic shot and hedge instead of guessing, and it is
the gate in the next section.

## Where it stops working

A shot's resting position is a smooth function of its inputs right up to the
point where the ball starts ricocheting. After a few cushion contacts, a
millimetre of difference in the cue position moves the final position by a table
length, and no model recovers that.

![Error by cushion contacts](assets/accuracy.png)

So the results are reported by cushion contact count rather than as a single
average. The residual model is the best of the three for shots with 0, 1 or 2
cushion contacts; past that, plain gradient boosting edges ahead by hedging
harder toward the middle of the table. The reported spread ratio makes that
visible: CueNet reproduces 97% of the true spread in the predictable buckets and
87% in the chaotic one.

## Reporting confidence without a second model

Slicing by the simulator's cushion count is an autopsy: you only have that number
once you have paid for the simulation. The closed-form solver, though, reports the
cushion count it *expects* while computing its answer, so that signal is free —
and it is a good enough proxy for "is this outcome chaotic" to gate on:

| Answer only when the solver expects | Coverage | CueNet error |
|---|---:|---:|
| No cushion | 9.8% | 100 mm |
| At most one | 27.5% | 189 mm |
| At most two | 50.0% | 253 mm |
| Anything | 100% | 378 mm |

![Error against coverage](assets/reliability.png)

This is what makes the fast path deployable rather than merely fast: a caller can
choose an error budget, take the coverage that comes with it, and route the
remainder to the simulator. The alternative — a single headline error over a
distribution that mixes 100 mm shots with 700 mm shots — gives a caller no way to
act. The residual model happens to be the most accurate of the three at every
coverage level, which is a more robust claim than its 1.9% edge on the mean.

## What the surrogate is and is not good enough for

Worth being precise, because "0.58 ms billiards predictor" invites the wrong
conclusion.

**Not accurate enough to aim with.** On a representative layout — cue ball at
(0.60, 0.35), object ball at (1.70, 0.75), far corner pocket — the aim window
that actually pots the ball is **0.15° to 0.25° wide**, at stroke speeds of 1.5,
2.5 and 4 m/s. Ranking thousands of candidate aim lines by the surrogate's
predicted object endpoint does concentrate them near the correct angle, but the
top-ranked candidates spread over roughly two degrees, so the ranking never
resolves a window an order of magnitude narrower than that.

This is less of a loss than it looks, because aiming needs no learned model at
all. The contact geometry is exact and closed-form: the ghost-ball point is the
object ball's centre displaced one diameter back along the line to the pocket, and
shooting at it pots the ball at every speed tested — which
`tests/test_validation.py` asserts, along with half a degree off it missing. The
window is not symmetric
about it, though — it extends about 0.1° to the thin side and barely at all to the
thick side, which is collision-induced throw, the object ball being dragged off
the geometric line by ball-ball friction. That the simulator reproduces an effect
real players compensate for by feel is a better argument for the cloth model than
any of the aggregate error numbers.

**Weakest exactly where the baseline is weakest.** Two thirds of sampled shots
never reach the object ball, and for those the baseline's "it stays put" is exactly
right, so the reported object-ball error of 239 mm is mostly an average over free
zeros. On the third of shots that do involve a collision, the numbers are 738 mm
for the cue ball and 701 mm for the object ball — and plain gradient boosting beats
the residual model there, 638 mm and 606 mm.

That is a structural consequence of the design rather than a tuning problem. The
closed-form solver has no ball-ball contact model, so on a collision it can be a
metre wrong, and the residual is then asked to undo a large error rather than
refine a small one. A model predicting endpoints directly has no such anchor to
fight. The corollary is that the highest-value next change is not a bigger network
or more data: it is giving the closed-form solver a ghost-ball collision so the
baseline it hands over is worth correcting.

The residual formulation wins the other side of that trade. On shots where nothing
happens it introduces 18 mm of spurious object-ball motion against gradient
boosting's 76 mm, because "predict no correction" is where it starts rather than
something it has to infer.

So the honest summary: a good surrogate for the *distribution* of low-cushion
outcomes and for screening large candidate sets, an exact tool for contact
geometry, weaker than a plain regressor once a collision is involved, and not a
substitute for the simulator when a specific multi-rail outcome matters.

## Choices a reviewer might question

**Explicit Euler at 1 ms rather than something higher order.** The dynamics are
piecewise smooth with impulsive events at contacts, so the local error is
dominated by event handling, not by integrator order. The validation suite bounds
the resulting drift at under 1% against closed form. A symplectic or adaptive
integrator would be the right call if the cloth model were stiffer.

**Positional correction on overlap.** Overlapping balls are pushed apart before
the impulse is applied, which is a projection rather than a physical force. It is
what keeps a packed rack stable; the alternative is sub-stepping to the exact
contact time, which costs more than the accuracy is worth here.

**Endpoints as the learning target.** Resting position is what shot selection
needs. Predicting whole trajectories would be a sequence-model problem, and the
chaos analysis above suggests the horizon over which that is worth attempting is
short.

**Dataset size.** 20,000 shots, which is about fifteen minutes of generation on
eight cores. No learning curve was measured, so this is a practical choice rather
than a demonstrated sufficiency — the honest expectation is that more data would
help the collision cases, where the model is fighting the baseline, and do very
little for the multi-rail cases, where the target is chaotic. Generation is
parallel and seeded per sample, so the dataset is identical on 1 core or 32.
