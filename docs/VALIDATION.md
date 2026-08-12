# Validation

The simulator is checked against closed-form solutions and conservation laws
rather than against its own previous output, so a regression means the physics is
wrong and not merely different. Everything below is asserted in
`tests/test_validation.py` and runs in about ten seconds.

```bash
pytest tests/test_validation.py -v
```

![Simulated versus closed-form sliding phase](assets/validation.png)

## The cloth model

A ball on cloth is in one of four states, decided by the velocity of its contact
point, `u = v + ω × (-Rẑ)`:

| State | Condition | Dynamics |
|---|---|---|
| Sliding | `u ≠ 0` | friction `μ_s g` opposite `u`, with the matching torque |
| Rolling | `u = 0` | rolling resistance `μ_r g` opposite `v` |
| Spinning | `v = 0`, `ω_z ≠ 0` | `dω_z/dt = -5 μ_sp g / 2R` |
| Stationary | both zero | at rest |

While sliding, the slip velocity decays 3.5x faster than the centre of mass,
because friction both slows the ball and spins it up:

```
|du/dt| = μ_s g (1 + mR²/I) = μ_s g (1 + 5/2)
```

## What is checked

| Property | Reference | Tolerance | Measured |
|---|---|---|---|
| Speed when rolling begins | `5/7 v₀` | 1% | 0.3% |
| Sliding distance | `12v₀² / (49 μ_s g)` | 2% | 0.8% |
| Total stopping distance | slide + `v²/(2 μ_r g)` | 1% | 0.6% |
| Natural roll tip offset | no sliding phase at `f = 0.4` | exact | `u = 0` |
| Vertical spin decay | `5 μ_sp g / 2R` | 1e-6 | exact |
| Cushion rebound speed | `e · v_approach` | 1e-9 | exact |
| Momentum in a frictional collision | conserved | 1e-12 | exact |
| Energy in any contact | never increases | — | holds |
| Sidespin on a rolling ball | no lateral deflection | 1e-9 m | exact |
| Draw / stun / follow | ordered, draw ends behind contact | — | holds |
| Ghost-ball aim line | pots the ball; ±0.5° misses | — | holds |
| Timestep convergence, full-ball contact | 2 ms vs 1 ms | 10 mm | 3 mm |
| Timestep convergence, thin cut | 2 ms vs 1 ms | 60 mm | 46 mm |
| Contacts a racked triangle presents | 30 | exact | 30 |
| Balls set moving by a full-power break | all 15 | — | holds |

## The bug none of the above could see

Every property in that table concerns one ball, or two. A defect that lives in
the relationship between fifteen of them passes all of it, and one did.

Two balls were treated as touching when the gap between their surfaces was at
most `1e-4 m`. The rack was built with a `1e-4 m` clearance. So all thirty
contacts in the triangle sat exactly on the threshold that decides whether a
contact exists, and which side each one landed on was decided by whether
`hypot` rounded up or down for that particular pair. Sixteen registered.
Fourteen did not.

The consequences were visible only in aggregate. A break propagated through a
contact graph with holes in it, so balls in the middle of the rack came out of
a full-power break having barely moved, and mean pairwise separation after a
break *fell* as cue speed rose — the table opened up less the harder it was
struck:

| Cue speed | 3 m/s | 5 m/s | 6.5 m/s | 8.2 m/s | 10 m/s |
|---|---:|---:|---:|---:|---:|
| Before, mean pair separation | 0.32 m | 0.57 m | 0.53 m | 0.46 m | 0.51 m |
| After | 0.31 m | 0.62 m | 0.59 m | 0.54 m | 0.56 m |

Averaged over 40 racks each. It was found by measuring that relationship and
getting the sign wrong, not by a test going red — and not by the browser parity
harness either, which reproduced the broken contact graph to eleven decimal
places because it was a faithful port of it. Two implementations agreeing is
evidence about the port, and about nothing else.

The tolerance now has a name, `CONTACT_BAND`, the broad and narrow phases use
the same one, and the balls are racked touching so that nothing sits on the
boundary. Tests assert all three, including that the band stays far from both
the floating-point noise that would swallow it and the physical scale that
would make it meaningless.

## Is the reference converged?

Worth asking, because everything downstream is measured against it. Training
labels are generated at a 2 ms step, so if that step were too coarse the reported
model errors would partly be measuring the integrator.

Halving the step moves the resting position by about 3 mm on a full-ball contact
and 46 mm on a thin cut, the thin cut being the worst case because a small change
in contact geometry is amplified into a large change in direction. Quartering it,
to 0.5 ms, moves free-ball endpoints by a median of 0.8 mm and at most 19 mm.
Against the 100 mm error the surrogate achieves on its best shots, discretisation
is a real but sub-dominant term — and it is bounded by a test rather than assumed.

The tip offset result is worth spelling out because it is easy to get wrong. A
horizontal impulse applied a distance `d = f·R` off centre gives `Δv = J/m` and
`Δω = J d / I`, so with `I = (2/5)mR²`:

```
ω = 2.5 · f · v / R
```

Pure rolling requires `ω = v/R`, so `f = 0.4`: a tip `2R/5` above centre, `7R/5`
above the cloth, launches the ball already rolling. Offsets beyond `|f| = 0.5`
are past the practical miscue limit, which is where the dataset sampling stops.

![Draw, stun and follow from the same stroke speed](assets/spin_control.png)

Draw deserves a note as well. A ball struck with backspin on an open table does
**not** come back: friction removes the backspin before it removes the forward
velocity, and it ends up rolling forward. Draw only works because a full-ball
collision takes the forward velocity away while the backspin survives. The test
therefore asserts the ordering across a contact, not the behaviour of a free
ball, and the figure above shows the cue ball finishing 0.18 m up-table, well
behind the 1.14 m contact point.

## What is not modelled

Stated plainly, because a validation document that only lists successes is not
worth much:

- **Cue ball squirt and swerve.** A real off-centre hit deflects the cue ball a
  degree or two off the aim line and then curves it. Neither is modelled, so
  aiming corrections from this simulator would be systematically wrong for
  heavy english.
- **Massé.** Cue elevation adds a little vertical-axis spin and nothing else.
- **Cushion geometry.** Rails are treated as flat vertical planes at the ball
  centre height, with a speed-dependent restitution as a stand-in for cushion
  compliance. Real cushions contact above centre and lift the ball slightly.
- **Pocket geometry.** A pocket is a capture radius, not a jaw. Balls that would
  rattle out in reality drop here.
- **Cloth inhomogeneity** is a smooth analytic noise field, not a measured one.
- **Ball-ball throw** uses a velocity-dependent friction coefficient of the
  usual form `μ_b ≈ a + b·exp(-c|v_rel|)`, with coefficients that are plausible
  rather than fitted to measurements.

None of these are calibrated against a real table, because no measurements were
taken. The claim this project makes is internal consistency with classical
mechanics, not fidelity to a specific piece of equipment.
