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
