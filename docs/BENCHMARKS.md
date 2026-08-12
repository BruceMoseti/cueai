# Benchmarks

Regenerate with `python scripts/benchmark.py`. Single CPU core, no GPU.

- Python 3.12.3 on Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64

## Latency per shot

| Method | Mean | Median | p95 | Speedup vs full rack |
|---|---:|---:|---:|---:|
| Numerical simulator, 16-ball rack | 36.8 s | 36.8 s | 36.9 s | — |
| Numerical simulator, cue + object ball | 350.0 ms | 341.3 ms | 380.3 ms | 105x |
| Closed-form solver (no ML) | 0.261 ms | 0.259 ms | 0.269 ms | 141,228x |
| Gradient boosting on the same features | 2.6 ms | 2.6 ms | 2.7 ms | 13,948x |
| Closed form + CueNet residual (ONNX Runtime) | 0.626 ms | 0.571 ms | 0.595 ms | 58,809x |
| CueNet forward pass only, batch of 1024 (per shot) | 1.3 us | 0.8 us | 4.7 us | — |

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
