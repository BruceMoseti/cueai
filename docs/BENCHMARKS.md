# Benchmarks

Regenerate with `python scripts/benchmark.py`. Single CPU core, no GPU.

- Python 3.12.3 on Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64

## Latency per shot

| Method | Mean | Median | p95 | Speedup vs full rack |
|---|---:|---:|---:|---:|
| Numerical simulator, 16-ball rack | 35.2 s | 35.4 s | 35.5 s | — |
| Numerical simulator, cue + object ball | 333.9 ms | 333.8 ms | 335.2 ms | 105x |
| Closed-form solver (no ML) | 0.256 ms | 0.254 ms | 0.264 ms | 137,489x |
| Gradient boosting on the same features | 2.5 ms | 2.5 ms | 2.6 ms | 13,997x |
| Closed form + CueNet residual (ONNX Runtime) | 0.568 ms | 0.563 ms | 0.583 ms | 61,985x |
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

## Knowing which predictions to trust

The table above slices by what the simulator did, which is only knowable after paying for the simulation. This one slices by the cushion count the closed-form solver *expects*, which is already computed as part of making the prediction and therefore costs nothing extra. Answering only the shots below a threshold trades coverage for accuracy:

| Expected cushions | Shots answered | Coverage | Closed form | Gradient boosting | CueNet residual |
|---|---:|---:|---:|---:|---:|
| ≤ 0 | 390 | 9.8% | 113 mm | 158 mm | 100 mm |
| ≤ 1 | 1098 | 27.5% | 240 mm | 220 mm | 189 mm |
| ≤ 2 | 2002 | 50.0% | 334 mm | 266 mm | 253 mm |
| ≤ 3 | 2748 | 68.7% | 401 mm | 316 mm | 306 mm |
| no gate | 4000 | 100.0% | 494 mm | 385 mm | 378 mm |
