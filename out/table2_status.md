# Table II - status of each row

| Row | Status in this package | How to obtain the final number |
|---|---|---|
| Verification errors | **Not measured** (LFW hosts blocked in the cloud session). Script + unit tests ready. | `python identity_eval.py --download --model mbf --out t2_mbf` |
| Identification errors | **Not measured** (same reason). Script + unit tests ready. | same command |
| Acquisition failures | **Not measured** on LFW. Detector/quality-gate verified on a bundled test image (face accepted; 3 non-face images rejected as `no_face`). | same command |
| Latency | **Measured in the cloud sandbox as a pipeline check only** (table below). Not target hardware. | run the command on the laptop / Raspberry Pi / Jetson |
| Resource use | Peak RSS and model sizes measured in the sandbox (below); **energy not measured** (no RAPL, no power meter). | USB power meter + `--measured-energy-j` |
| Population coverage | Determined by dataset: LFW = adult public figures, no age/sex/skin-tone labels, no children. | printed by the script |

## Sandbox benchmark (cloud VM, x86_64, 2 vCPU, ONNX Runtime CPU, 200 attempts after 20 warm-up, fresh process)

| Pipeline | detect (median ms) | embed (median ms) | acquisition-to-decision, gallery 100: median / P95 (ms) | peak RSS (MB) | model files (MB) |
|---|---|---|---|---|---|
| MobileFaceNet (SCRFD-500M + w600k_mbf), 1 thread | 4.9 | 14.0 | 19.0 / 23.4 | 142 | 16.1 |
| MobileFaceNet, 2 threads | 4.8 | 17.4 | 22.4 / 31.9 | 142 | 16.1 |
| ArcFace-R50 (SCRFD-10G + w600k_r50), 1 thread | 54.3 | 237.4 | 291.3 / 321.1 | 460 | 191.3 |

The same image (a 512x512 portrait bundled with scikit-image) was used for every attempt, so these numbers reflect compute cost, not recognition accuracy.
