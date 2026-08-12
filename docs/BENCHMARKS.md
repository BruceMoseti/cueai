# Benchmarks

Regenerate with `python scripts/benchmark.py`. Single CPU core, no GPU.

- Python 3.12.3 on Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64

## Latency per shot

| Method | Mean | Median | p95 | Speedup vs full rack |
|---|---:|---:|---:|---:|
| Numerical simulator, 16-ball rack | 4.6 s | 4.6 s | 4.6 s | — |
| Numerical simulator, cue + object ball | 337.9 ms | 337.8 ms | 338.7 ms | 14x |
| Closed-form solver (no ML) | 0.266 ms | 0.264 ms | 0.275 ms | 17,290x |
| Gradient boosting on the same features | 2.6 ms | 2.6 ms | 2.8 ms | 1,744x |
| Closed form + CueNet residual (ONNX Runtime) | 0.602 ms | 0.572 ms | 0.590 ms | 7,638x |
| CueNet forward pass only, batch of 1024 (per shot) | 1.6 us | 0.8 us | 4.7 us | — |

## Accuracy on held-out shots

4,000 test shots from a 20,000 shot dataset. Error is the distance between predicted and simulated resting position, averaged over the cue ball and the object ball.

| Model | Mean error | p95 | Cue ball | Object ball | R² |
|---|---:|---:|---:|---:|---:|
| Closed form (no fitting) | 494 mm | 1778 mm | 730 mm | 258 mm | 0.013 |
| Gradient boosting on raw features | 382 mm | 1163 mm | 516 mm | 249 mm | 0.521 |
| Closed form + CueNet residual | 376 mm | 1293 mm | 516 mm | 236 mm | 0.431 |

## Where prediction stops working

Resting position is a smooth function of the shot until the ball starts ricocheting between cushions. Mean error in mm by cushion contacts:

| Cushion contacts | Shots | Closed form | Gradient boosting | CueNet residual |
|---|---:|---:|---:|---:|
| 0 | 414 | 114 | 162 | 98 |
| 1 | 650 | 225 | 222 | 187 |
| 2 | 876 | 371 | 289 | 281 |
| 3+ | 2060 | 708 | 517 | 532 |

## Reading the object-ball number honestly

Only about a third of sampled shots actually hit the object ball. For the rest it stays where it started, which the closed-form baseline predicts exactly, so those rows contribute almost no object-ball error to any model. Split out, in mm:

| Ball-ball contact | Shots | Share | Closed form cue / object | Gradient boosting cue / object | CueNet cue / object |
|---|---:|---:|---:|---:|---:|
| no | 2705 | 67.6% | 585 / 0 | 457 / 77 | 411 / 16 |
| yes | 1295 | 32.4% | 1033 / 798 | 638 / 610 | 736 / 696 |

## Feature ablation

Same architecture, epochs, seed and split; the ablated model sees only raw shot parameters, with none of the closed-form solver's conclusions. This is the measurement behind the claim that the features carry this result:

| CueNet inputs | All shots | Direct shots (no cushion) |
|---|---:|---:|
| Raw shot parameters only | 469 mm | 179 mm |
| Plus closed-form output | 376 mm | 98 mm |
| _closed form alone, for reference_ | 494 mm | 114 mm |

## Knowing which predictions to trust

The cushion-contact breakdown above slices by what the simulator did, which is only knowable after paying for the simulation. This table slices by the cushion count the closed-form solver *expects*, which is already computed as part of making the prediction and therefore costs nothing extra. Answering only the shots below a threshold trades coverage for accuracy:

| Expected cushions | Shots answered | Coverage | Closed form | Gradient boosting | CueNet residual |
|---|---:|---:|---:|---:|---:|
| ≤ 0 | 390 | 9.8% | 113 mm | 160 mm | 100 mm |
| ≤ 1 | 1098 | 27.5% | 240 mm | 221 mm | 189 mm |
| ≤ 2 | 2002 | 50.0% | 334 mm | 265 mm | 251 mm |
| ≤ 3 | 2748 | 68.7% | 401 mm | 314 mm | 303 mm |
| no gate | 4000 | 100.0% | 494 mm | 382 mm | 376 mm |
