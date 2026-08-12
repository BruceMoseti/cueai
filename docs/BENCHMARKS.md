# Benchmarks

Regenerate with `python scripts/benchmark.py`. Single CPU core, no GPU.

- Python 3.12.3 on Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64

## Latency per shot

| Method | Mean | Median | p95 | Speedup vs full rack |
|---|---:|---:|---:|---:|
| Numerical simulator, 16-ball rack | 36.8 s | 36.7 s | 37.2 s | — |
| Numerical simulator, cue + object ball | 345.1 ms | 345.1 ms | 345.7 ms | 107x |
| Closed-form solver (no ML) | 0.261 ms | 0.259 ms | 0.270 ms | 140,996x |
| Gradient boosting on the same features | 2.5 ms | 2.5 ms | 2.6 ms | 14,454x |
| Closed form + CueNet residual (ONNX Runtime) | 0.584 ms | 0.571 ms | 0.593 ms | 63,032x |
| CueNet forward pass only, batch of 1024 (per shot) | 0.6 us | 0.6 us | 0.7 us | — |

## Accuracy on held-out shots

4,000 test shots from a 20,000 shot dataset. Error is the distance between predicted and simulated resting position, averaged over the cue ball and the object ball.

| Model | Mean error | p95 | Cue ball | Object ball | R² |
|---|---:|---:|---:|---:|---:|
| Closed form (no fitting) | 494 mm | 1780 mm | 730 mm | 258 mm | 0.013 |
| Gradient boosting on raw features | 385 mm | 1188 mm | 523 mm | 248 mm | 0.514 |
| Closed form + CueNet residual | 378 mm | 1336 mm | 517 mm | 239 mm | 0.423 |

## Where prediction stops working

Resting position is a smooth function of the shot until the ball starts ricocheting between cushions. Mean error in mm by cushion contacts:

| Cushion contacts | Shots | Closed form | Gradient boosting | CueNet residual |
|---|---:|---:|---:|---:|
| 0 | 414 | 114 | 160 | 97 |
| 1 | 650 | 225 | 223 | 193 |
| 2 | 875 | 370 | 291 | 288 |
| 3+ | 2061 | 708 | 522 | 531 |

## Reading the object-ball number honestly

Only about a third of sampled shots actually hit the object ball. For the rest it stays where it started, which the closed-form baseline predicts exactly, so those rows contribute almost no object-ball error to any model. Split out, in mm:

| Ball-ball contact | Shots | Share | Closed form cue / object | Gradient boosting cue / object | CueNet cue / object |
|---|---:|---:|---:|---:|---:|
| no | 2704 | 67.6% | 585 / 0 | 468 / 76 | 411 / 18 |
| yes | 1296 | 32.4% | 1032 / 797 | 638 / 606 | 738 / 701 |

## Feature ablation

Same architecture, epochs, seed and split; the ablated model sees only raw shot parameters, with none of the closed-form solver's conclusions. This is the measurement behind the claim that the features carry this result:

| CueNet inputs | All shots | Direct shots (no cushion) |
|---|---:|---:|
| Raw shot parameters only | 473 mm | 173 mm |
| Plus closed-form output | 378 mm | 97 mm |
| _closed form alone, for reference_ | 494 mm | 114 mm |

## Knowing which predictions to trust

The cushion-contact breakdown above slices by what the simulator did, which is only knowable after paying for the simulation. This table slices by the cushion count the closed-form solver *expects*, which is already computed as part of making the prediction and therefore costs nothing extra. Answering only the shots below a threshold trades coverage for accuracy:

| Expected cushions | Shots answered | Coverage | Closed form | Gradient boosting | CueNet residual |
|---|---:|---:|---:|---:|---:|
| ≤ 0 | 390 | 9.8% | 113 mm | 158 mm | 100 mm |
| ≤ 1 | 1098 | 27.5% | 240 mm | 220 mm | 189 mm |
| ≤ 2 | 2002 | 50.0% | 334 mm | 266 mm | 253 mm |
| ≤ 3 | 2748 | 68.7% | 401 mm | 316 mm | 306 mm |
| no gate | 4000 | 100.0% | 494 mm | 385 mm | 378 mm |
