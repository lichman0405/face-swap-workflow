"""
infer.py
────────
InstantID face transfer: 将 source 真人照的人脸身份迁移到 target 铅笔淡彩画中。

使用 img2img 模式，最大程度保留 target 的线条、构图和艺术风格，同时将
source 人脸的身份特征（InsightFace embedding + 关键点）注入生成结果。

支持设备:
  --device cuda   → NVIDIA GPU，使用 float16，速度最快
  --device mps    → Apple Silicon，使用 float32（MPS float16 有兼容性问题）
  --device cpu    → 纯 CPU，使用 float32，速度最慢

用法示例:
  # 自动检测设备（cuda > mps > cpu）
  python infer.py --source source/my_face.jpg --target target/sketch.jpg

  # 显式指定 Apple Silicon GPU
  python infer.py --source source/my_face.jpg --target target/sketch.jpg --device mps

  # 显式指定 NVIDIA GPU，调整参数
  python infer.py \\
      --source source/my_face.jpg \\
      --target target/sketch.jpg \\
      --device cuda \\
      --strength 0.70 \\
      --ip-scale 0.85 \\
      --steps 30 \\
      --output output/result.png
"""

import argparse
import os
import sys
import time
from pathlib import Path

# MPS 内存优化：取消 MPS 分配器的默认上限（20 GB），允许使用更多统一内存
# 必须在 import torch 之前设置，否则 MPS 初始化后无效
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import cv2
import numpy as np
import torch
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from PIL import Image

from pipeline_stable_diffusion_xl_instantid_img2img import (
    StableDiffusionXLInstantIDImg2ImgPipeline,
    draw_kps,
)


# ── Default prompts ────────────────────────────────────────────────────────────
# Tuned for Korean traditional illustration style (hanbok, ink wash, pencil sketch).
# Override via --prompt if your target has a different art style.
DEFAULT_PROMPT = (
    "Korean traditional illustration, hanbok, pencil sketch, soft watercolor wash, "
    "delicate colored pencil, fine line art, ink drawing, "
    "detailed facial features, elegant portrait, high quality, masterpiece"
)
DEFAULT_NEGATIVE_PROMPT = (
    "ugly, deformed, noisy, blurry, low quality, bad anatomy, "
    "photorealistic, 3d render, western style, modern clothing, "
    "extra limbs, duplicate faces"
)

# ── Model paths ───────────────────────────────────────────────────────────────
CHECKPOINTS_DIR = "./checkpoints"
CONTROLNET_PATH = os.path.join(CHECKPOINTS_DIR, "ControlNetModel")
FACE_ADAPTER_PATH = os.path.join(CHECKPOINTS_DIR, "ip-adapter.bin")
FACE_ENCODER_ROOT = "./"          # antelopev2 must be at ./models/antelopev2/
BASE_MODEL_ID = "wangqixun/YamerMIX_v8"


# ── Device selection ──────────────────────────────────────────────────────────
def resolve_device(requested: str) -> torch.device:
    """
    Resolve the compute device.

    Priority when 'auto':  cuda  →  mps  →  cpu
    Explicit requests are honoured as-is (with a warning if unavailable).
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)
        # Sanity checks
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Check your CUDA installation or use --device mps / cpu."
            )
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but torch.backends.mps.is_available() is False. "
                "Requires Apple Silicon Mac with macOS 12.3+. Use --device cpu."
            )

    return device


def dtype_for_device(device: torch.device) -> torch.dtype:
    """
    Return the best dtype for the given device.

    CUDA  → float16  (faster, less VRAM)
    MPS   → float32  (MPS float16 has known issues with SDXL attention layers)
    CPU   → float32
    """
    return torch.float16 if device.type == "cuda" else torch.float32


def insightface_providers(device: torch.device) -> list:
    """
    InsightFace ONNX runtime execution providers.

    CUDA → CUDAExecutionProvider (GPU acceleration)
    MPS / CPU → CPUExecutionProvider (InsightFace has no MPS backend)
    """
    if device.type == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ── Face detection + embedding extraction ─────────────────────────────────────
def extract_face_info(image_pil: Image.Image, app: FaceAnalysis):
    """
    Detect the largest face in image_pil.
    Returns (embedding, kps_image) or raises RuntimeError if no face found.
    """
    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)

    if not faces:
        raise RuntimeError(
            "No face detected in the source image. "
            "Please use a clear, front-facing portrait photo."
        )

    # Use the largest detected face
    face = sorted(
        faces,
        key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
    )[-1]

    embedding = face["embedding"]
    kps_image = draw_kps(image_pil, face["kps"])

    return embedding, kps_image


# ── Pipeline builder ──────────────────────────────────────────────────────────
def build_pipeline(device: torch.device, dtype: torch.dtype):
    """
    Load ControlNet + SDXL img2img pipeline and move to device.
    """
    print(f"  Loading ControlNet from {CONTROLNET_PATH} …")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_PATH,
        torch_dtype=dtype,
    )

    print(f"  Loading base model {BASE_MODEL_ID} …")
    pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
    )

    print(f"  Moving pipeline to device: {device} …")
    pipe.to(device)

    # Load IP-Adapter face weights
    print(f"  Loading IP-Adapter from {FACE_ADAPTER_PATH} …")
    pipe.load_ip_adapter_instantid(FACE_ADAPTER_PATH)

    # Memory optimisation — helpful on both MPS and CUDA with limited VRAM
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()  # 按 slice 计算 attention，减少峰值显存

    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(
    pipe,
    face_embedding: np.ndarray,
    kps_image: Image.Image,
    target_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    ip_scale: float,
    controlnet_scale: float,
    steps: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
) -> Image.Image:
    """
    Run img2img pipeline with InstantID face identity injection.
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    # Set IP-Adapter scale
    pipe.set_ip_adapter_scale(ip_scale)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=target_image,          # target sketch → img2img base
        control_image=kps_image,     # face keypoints → ControlNet condition
        image_embeds=face_embedding, # face id embedding → IP-Adapter
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    ).images[0]

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InstantID: transfer a real face identity onto a pencil sketch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--source", required=True, help="Path to source face image (real photo)")
    p.add_argument("--target", required=True, help="Path to target image (pencil sketch)")

    # Output
    p.add_argument("--output", default="output/result.png", help="Output image path")

    # Device
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help=(
            "Compute device. "
            "'auto' picks cuda → mps → cpu automatically. "
            "'cuda' = NVIDIA GPU (float16). "
            "'mps' = Apple Silicon GPU (float32). "
            "'cpu' = CPU-only (float32, slow)."
        ),
    )

    # Generation parameters
    p.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help=(
            "img2img denoising strength [0.0–1.0]. "
            "Lower = more of the original sketch is preserved. "
            "Higher = more freedom for face identity injection. "
            "Use 0.50–0.65 when the face is large in the target; "
            "0.70–0.85 when the face is small or side-facing."
        ),
    )
    p.add_argument(
        "--ip-scale",
        type=float,
        default=0.80,
        help=(
            "IP-Adapter face identity weight [0.0–1.0]. "
            "Higher = stronger face resemblance to source. "
            "Lower = more style/text control."
        ),
    )
    p.add_argument(
        "--controlnet-scale",
        type=float,
        default=0.80,
        help="ControlNet (face keypoints) conditioning scale [0.0–1.0].",
    )
    p.add_argument("--steps", type=int, default=30, help="Number of denoising steps.")
    p.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt describing the desired output style.")
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative text prompt.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    sep = f"{BOLD}{CYAN}{'─' * 60}{RESET}"

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    for flag, path in [("--source", args.source), ("--target", args.target)]:
        if not os.path.isfile(path):
            print(f"Error: {flag} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if not os.path.isdir(CONTROLNET_PATH):
        print(
            f"Error: ControlNet not found at {CONTROLNET_PATH}\n"
            f"Run:  python download_models.py",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(FACE_ADAPTER_PATH):
        print(
            f"Error: IP-Adapter not found at {FACE_ADAPTER_PATH}\n"
            f"Run:  python download_models.py",
            file=sys.stderr,
        )
        sys.exit(1)

    antelopev2_dir = os.path.join("models", "antelopev2")
    if not os.path.isdir(antelopev2_dir):
        print(
            f"Error: antelopev2 face encoder not found at {antelopev2_dir}\n"
            f"Run:   python download_models.py  (and follow the manual step)",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Device resolution ──────────────────────────────────────────────────────
    print(sep)
    device = resolve_device(args.device)
    dtype = dtype_for_device(device)
    providers = insightface_providers(device)

    print(f"  {BOLD}Device    :{RESET}  {device}  ({dtype})")
    print(f"  {BOLD}Source    :{RESET}  {args.source}")
    print(f"  {BOLD}Target    :{RESET}  {args.target}")
    print(f"  {BOLD}Output    :{RESET}  {args.output}")
    print(f"  {BOLD}Strength  :{RESET}  {args.strength}  (img2img denoising)")
    print(f"  {BOLD}IP-scale  :{RESET}  {args.ip_scale}  (face identity weight)")
    print(f"  {BOLD}Steps     :{RESET}  {args.steps}")
    print(f"  {BOLD}Seed      :{RESET}  {args.seed}")
    print(sep)

    # ── Load images ────────────────────────────────────────────────────────────
    source_image = Image.open(args.source).convert("RGB")
    target_image = Image.open(args.target).convert("RGB")

    # Resize target to a stable SDXL resolution (multiple of 64)
    w, h = target_image.size
    scale = 1024 / max(w, h)
    new_w = (int(w * scale) // 64) * 64
    new_h = (int(h * scale) // 64) * 64
    target_image = target_image.resize((new_w, new_h), Image.LANCZOS)
    print(f"  Target resized to: {new_w} × {new_h}")

    # ── Face encoder ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  {BOLD}[1/3] Loading face encoder (InsightFace antelopev2) …{RESET}")
    print(sep)

    app = FaceAnalysis(
        name="antelopev2",
        root=FACE_ENCODER_ROOT,
        providers=providers,
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("  Extracting face embedding from source image …")
    face_embedding, kps_image = extract_face_info(source_image, app)
    print(f"  {GREEN}Face detected. Embedding shape: {face_embedding.shape}{RESET}")

    # ControlNet 要求 control_image 与 target_image 等尺寸
    kps_image = kps_image.resize(target_image.size, Image.LANCZOS)
    print(f"  kps_image resized to: {kps_image.size[0]} × {kps_image.size[1]}")

    # ── Load pipeline ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  {BOLD}[2/3] Loading diffusion pipeline …{RESET}")
    print(sep)

    t0 = time.time()
    pipe = build_pipeline(device, dtype)
    print(f"  {GREEN}Pipeline ready in {time.time() - t0:.1f}s{RESET}")

    # ── Inference ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  {BOLD}[3/3] Running inference …{RESET}")
    print(sep)

    t1 = time.time()
    result = run_inference(
        pipe=pipe,
        face_embedding=face_embedding,
        kps_image=kps_image,
        target_image=target_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        ip_scale=args.ip_scale,
        controlnet_scale=args.controlnet_scale,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device,
    )
    elapsed = time.time() - t1

    result.save(args.output)
    print(f"\n  {GREEN}{BOLD}Done!{RESET}  Inference took {elapsed:.1f}s")
    print(f"  Output saved → {os.path.abspath(args.output)}")
    print(sep + "\n")


if __name__ == "__main__":
    main()
