# Design

## The problem this solves

Simulating a billiards shot accurately is slow. Sixteen bodies, each in one of
four motion states, with collisions resolved every millisecond, costs about
**36 seconds per rack shot** in this implementation. Anything that needs many
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
              numerical simulator                36 s, ground truth
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
closed-form solver on the easiest shots: 191 mm against the baseline's 114 mm for
shots that never touch a cushion. The network was being fed raw shot parameters,
so to know where the closed-form model went wrong it first had to rediscover
cushion reflection geometry from a speed and an angle. It spent its capacity on
the chaotic shots that dominate the loss and degraded the clean ones.

The fix was to hand the network what the physics already knew: the closed-form
endpoint, the cushion count it expects, whether it expects a pot, and the
ghost-ball geometry deciding whether the object ball is contacted at all. Same
architecture, same data, same epochs — the error on those clean shots went from
191 mm to 97 mm, below the baseline. The predicted cushion count also lets the
model recognise a chaotic shot and hedge instead of guessing.

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

The honest summary is that this is a good surrogate for the shots a player
actually aims — direct pots, position play, one rail — and a poor predictor of
multi-rail scatter, which is a property of the physics rather than of the model.

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

**Dataset size.** 20,000 shots, chosen because the error curves had flattened,
not because it was the largest number available. Generation is parallel and
seeded per sample, so the dataset is identical on 1 core or 32.
