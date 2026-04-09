# InstantID Face Transfer

Transfer the facial identity from a real portrait photo into a pencil sketch or watercolor illustration — preserving the original artwork's lines, composition, and style while making the subject resemble the source person.

**Platforms: Windows (NVIDIA / CPU) · macOS Apple Silicon (MPS) · Linux (NVIDIA / CPU)**

---

## How It Works

```
source/  →  InsightFace extracts face embedding + keypoints
target/  →  img2img base image (preserves art style & composition)
               │
               ▼
     SDXL + IP-Adapter (identity injection)
           + ControlNet (keypoint alignment)
               │
               ▼
output/  →  Illustration with transferred facial identity
```

Core components:
- **InsightFace antelopev2** — face detection and 512-dim identity embedding extraction
- **InstantID IP-Adapter** — injects face embedding into SDXL cross-attention layers
- **InstantID ControlNet** — constrains face pose and position via facial keypoints
- **SDXL img2img** — denoises from the target sketch, fusing facial identity

---

## Project Structure

```
deep-fake-face/
├── source/                         # Source portrait photos (real person)
├── target/                         # Target artwork (pencil sketch / illustration)
├── output/                         # Inference output directory
│
├── checkpoints/                    # InstantID model weights (auto-downloaded)
│   ├── ControlNetModel/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   └── ip-adapter.bin
│
├── models/
│   └── antelopev2/                 # InsightFace encoder (manual download required)
│       ├── 1k3d68.onnx
│       ├── 2d106det.onnx
│       ├── genderage.onnx
│       ├── glintr100.onnx
│       └── scrfd_10g_bnkps.onnx
│
├── ip_adapter/                     # InstantID IP-Adapter implementation
├── pipeline_stable_diffusion_xl_instantid_img2img.py
│
├── infer.py                        # Main inference script
├── download_models.py              # Model download script
├── requirements.txt                # Python dependencies
└── setup_env.py                    # Cross-platform environment setup script
```

---

## Environment Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda recommended)
- NVIDIA users: driver ≥ 525, CUDA 12.1+
- Apple Silicon users: macOS 12.3+

### One-Command Setup (Recommended — All Platforms)

`setup_env.py` is a pure-Python script that runs on **Windows / macOS / Linux**. It automatically detects your platform and CUDA version, then installs the matching PyTorch build.

```bash
# Auto-detect platform and CUDA version (recommended)
python setup_env.py

# Force a specific CUDA build (use if auto-detection is wrong)
python setup_env.py --cu128   # CUDA 12.8+  — required for Blackwell RTX 5000 series
python setup_env.py --cu124   # CUDA 12.4–12.7
python setup_env.py --cu121   # CUDA 12.1–12.3
python setup_env.py --cpu     # CPU-only
```

> **Windows users:** Run inside **Anaconda Prompt** or a conda-initialized PowerShell. Do not use plain `cmd`.

### Manual Setup

#### Step 1 — Create conda environment

```bash
conda create -n instantid python=3.10 -y
conda activate instantid
```

> **Why Python 3.10?**
> The dependency chain (diffusers, insightface, onnxruntime) is most stable on 3.10. 3.11+ is not recommended.

#### Step 2 — Install PyTorch

PyTorch must be installed **before** other dependencies, as the command differs by platform:

| Platform | Command |
|----------|---------|
| Apple Silicon (MPS) | `pip install torch torchvision torchaudio` |
| NVIDIA CUDA 12.8 — RTX 5000 (Blackwell) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| NVIDIA CUDA 12.4 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| NVIDIA CUDA 12.1 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| CPU only | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |

> **Windows note:** `pip install torch` installs the CPU build by default. You **must** add `--index-url` to get a GPU build.

#### Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Downloading Models

### Automatic (InstantID weights)

```bash
python download_models.py
```

If HuggingFace is slow or blocked, use the mirror flag:

```bash
python download_models.py --mirror
```

Downloads to:
- `checkpoints/ControlNetModel/` — InstantID ControlNet (~2.5 GB)
- `checkpoints/ip-adapter.bin` — InstantID IP-Adapter (~1.1 GB)

### Manual (antelopev2 face encoder)

InsightFace licensing restricts automatic distribution. Download manually:

1. Open: <https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304>
2. Download `antelopev2.zip`
3. Extract to `models/antelopev2/`:
   ```bash
   unzip antelopev2.zip -d models/antelopev2/
   ```

### SDXL Base Model

`wangqixun/YamerMIX_v8` (~6 GB) is downloaded and cached automatically from HuggingFace on the first run of `infer.py`.

---

## Usage

```bash
conda activate instantid
```

### Basic (auto-detect device)

```bash
python infer.py \
    --source source/my_face.jpg \
    --target target/sketch.jpg
```

Output is saved to `output/result.png` by default.

### Explicit Apple Silicon GPU

```bash
python infer.py \
    --source source/my_face.jpg \
    --target target/sketch.jpg \
    --device mps
```

### Explicit NVIDIA GPU

```bash
python infer.py \
    --source source/my_face.jpg \
    --target target/sketch.jpg \
    --device cuda
```

### Full Example

```bash
python infer.py \
    --source   source/my_face.jpg \
    --target   target/sketch.jpg \
    --output   output/result.png \
    --device   cuda \
    --strength 0.65 \
    --ip-scale 0.80 \
    --steps    30 \
    --seed     42
```

---

## Parameter Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | required | Source portrait path (clear frontal face recommended) |
| `--target` | str | required | Target artwork path (pencil sketch / illustration) |
| `--output` | str | `output/result.png` | Output image path |
| `--device` | auto/cuda/mps/cpu | `auto` | Compute device; auto selects cuda → mps → cpu |
| `--strength` | 0.0–1.0 | `0.75` | img2img denoising strength. **Lower = more sketch preserved.** Use 0.70–0.85 for small/angled faces; 0.50–0.65 for large faces |
| `--ip-scale` | 0.0–1.0 | `0.80` | Face identity weight. **Higher = more similar to source person** |
| `--controlnet-scale` | 0.0–1.0 | `0.80` | Facial keypoint ControlNet strength |
| `--steps` | int | `30` | Denoising steps. More = better quality, slower |
| `--guidance-scale` | float | `5.0` | Text guidance strength (CFG scale) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--prompt` | str | preset | Positive prompt describing the desired art style |
| `--negative-prompt` | str | preset | Negative prompt |

### Tuning Tips

**Sketch composition not preserved (result drifts from original artwork):**
```
--strength 0.50 --controlnet-scale 0.90
```

**Face not similar enough to source person:**
```
--ip-scale 0.90 --controlnet-scale 0.85
```

**Result looks too photorealistic, not enough sketch feel:**
Strengthen sketch keywords in `--prompt`: `pencil sketch, ink drawing, monochrome`

---

## Device Compatibility

| | Apple Silicon (MPS) | NVIDIA GPU (CUDA) | CPU |
|--|:--:|:--:|:--:|
| PyTorch dtype | float32 | float16 | float32 |
| InsightFace provider | CPUExecutionProvider | CUDAExecutionProvider | CPUExecutionProvider |
| Speed reference (30 steps) | ~60–120s | ~15–30s | ~600s+ |
| Memory requirement | Unified memory ≥ 16 GB | VRAM ≥ 12 GB | RAM ≥ 16 GB |
| `--device` value | `mps` | `cuda` | `cpu` |

> **MPS float32 note:** PyTorch's MPS backend has known issues with SDXL float16 attention layers. float32 is enforced to avoid numerical errors, at the cost of ~2× memory vs CUDA float16.

---

## FAQ

**`No face detected in the source image`**  
Ensure the source image contains a clear, large, frontal face. Resolution ≥ 512 px recommended. Avoid heavy occlusion.

**`RuntimeError: MPS backend out of memory`**  
Unified memory is exhausted. Add `pipe.enable_model_cpu_offload()` after pipeline creation in `infer.py`, or reduce `--steps`.

**Output lost all sketch style and looks like a real portrait**  
Lower `--strength` (e.g. 0.45) and `--ip-scale` (e.g. 0.65). Strengthen sketch keywords in `--prompt`.

**Output person does not resemble the source photo**  
Increase `--ip-scale` (e.g. 0.90) and `--controlnet-scale` (e.g. 0.90).

**CUDA out of memory (OOM)**  
Reduce `--steps`, or set the environment variable before running:
```bash
# Linux / macOS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Windows (PowerShell)
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

**HuggingFace download is slow or blocked**  
Use `python download_models.py --mirror`, or set:
```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows (PowerShell)
$env:HF_ENDPOINT="https://hf-mirror.com"
```

---

## License

- InstantID code: [Apache 2.0 License](https://github.com/instantX-research/InstantID/blob/main/LICENSE)
- InsightFace models (antelopev2): **non-commercial research use only** — see [InsightFace License](https://github.com/deepinsight/insightface?tab=readme-ov-file#license)
- InstantID checkpoints: **research use only**

Please ensure your use complies with applicable laws and respects individuals' portrait rights.
