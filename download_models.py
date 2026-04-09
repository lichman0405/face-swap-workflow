"""
download_models.py
──────────────────
Download all model weights required by InstantID face transfer.

Models downloaded:
  1. InstantID ControlNetModel  → ./checkpoints/ControlNetModel/
  2. InstantID ip-adapter.bin   → ./checkpoints/ip-adapter.bin
  3. antelopev2 face encoder    → ./models/antelopev2/   (manual, see note)

The SDXL base model (wangqixun/YamerMIX_v8, ~6 GB) will be downloaded
automatically the first time you run infer.py via the HuggingFace cache.

Usage:
    python download_models.py
    python download_models.py --mirror   # use hf-mirror.com (China mainland)
"""

import argparse
import os
import sys

from huggingface_hub import hf_hub_download


# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"
BOLD = "\033[1m"


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {YELLOW}→{RESET}  {msg}")


# ── Download helpers ───────────────────────────────────────────────────────────
def download_instantid_checkpoints(local_dir: str) -> None:
    """Download InstantID ControlNet + IP-Adapter weights."""
    repo = "InstantX/InstantID"
    files = [
        "ControlNetModel/config.json",
        "ControlNetModel/diffusion_pytorch_model.safetensors",
        "ip-adapter.bin",
    ]

    os.makedirs(local_dir, exist_ok=True)

    for filename in files:
        dest = os.path.join(local_dir, filename)
        if os.path.exists(dest):
            ok(f"Already exists, skipping: {filename}")
            continue
        info(f"Downloading {filename} ...")
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        ok(f"Saved → {dest}")


# ── Manual download note ───────────────────────────────────────────────────────
ANTELOPEV2_NOTE = f"""
{BOLD}{YELLOW}  Manual download required — antelopev2{RESET}

  The InsightFace antelopev2 model cannot be downloaded automatically.
  Please follow these steps:

  1. Open the link below in your browser:
     https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304

  2. Download the file:  antelopev2.zip

  3. Extract it into:
     {os.path.abspath('models/antelopev2/')}

  Expected structure after extraction:
     models/antelopev2/
     ├── 1k3d68.onnx
     ├── 2d106det.onnx
     ├── genderage.onnx
     ├── glintr100.onnx
     └── scrfd_10g_bnkps.onnx
"""


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Download InstantID model weights")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Use hf-mirror.com (recommended for China mainland users)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="./checkpoints",
        help="Directory to save InstantID checkpoints (default: ./checkpoints)",
    )
    args = parser.parse_args()

    # Apply HuggingFace mirror if requested
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        info("Using HuggingFace mirror: https://hf-mirror.com")

    # ── Step 1: InstantID checkpoints ─────────────────────────────────────────
    section("Step 1 / 2  —  InstantID ControlNet + IP-Adapter")
    download_instantid_checkpoints(args.checkpoints_dir)

    # ── Step 2: antelopev2 (manual) ───────────────────────────────────────────
    section("Step 2 / 2  —  antelopev2 Face Encoder (manual)")
    antelopev2_dir = os.path.join("models", "antelopev2")
    if os.path.isdir(antelopev2_dir) and len(os.listdir(antelopev2_dir)) >= 5:
        ok("antelopev2 models already present.")
    else:
        print(ANTELOPEV2_NOTE)

    # ── Done ──────────────────────────────────────────────────────────────────
    section("Done")
    print(
        f"\n  {GREEN}All automatic downloads complete.{RESET}\n"
        f"  The SDXL base model ({BOLD}wangqixun/YamerMIX_v8{RESET}, ~6 GB)\n"
        f"  will be downloaded automatically on first run of infer.py.\n"
    )


if __name__ == "__main__":
    main()
