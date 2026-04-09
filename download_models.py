"""
download_models.py
──────────────────
Download all model weights required by InstantID face transfer.

Models downloaded:
  1. InstantID ControlNetModel  → ./checkpoints/ControlNetModel/
  2. InstantID ip-adapter.bin   → ./checkpoints/ip-adapter.bin
  3. antelopev2 face encoder    → ./models/antelopev2/   (auto, with manual fallback)

The SDXL base model (wangqixun/YamerMIX_v8, ~6 GB) will be downloaded
automatically the first time you run infer.py via the HuggingFace cache.

Usage:
    python download_models.py
    python download_models.py --mirror   # use hf-mirror.com (China mainland)
"""

import argparse
import os
import sys
import urllib.request
import zipfile

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


# ── antelopev2 auto-download ───────────────────────────────────────────────────

ANTELOPEV2_URL = "http://storage.insightface.ai/files/models/antelopev2.zip"
ANTELOPEV2_EXPECTED_FILES = {
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx",
}
ANTELOPEV2_MANUAL_NOTE = f"""
{BOLD}{YELLOW}  Manual download required — antelopev2{RESET}

  The InsightFace server is currently unreachable. Please download manually:

  1. Open the link below in your browser:
     https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304

  2. Download the file:  antelopev2.zip

  3. Extract it into:
     {os.path.abspath('models/antelopev2/')}

  Expected files after extraction:
     models/antelopev2/1k3d68.onnx
     models/antelopev2/2d106det.onnx
     models/antelopev2/genderage.onnx
     models/antelopev2/glintr100.onnx
     models/antelopev2/scrfd_10g_bnkps.onnx
"""


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)


def download_antelopev2(dest_dir: str) -> bool:
    """
    Try to auto-download antelopev2.zip from the InsightFace official server.
    Returns True on success, False on failure.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Check if all files are already present
    existing = {f for f in os.listdir(dest_dir)} if os.path.isdir(dest_dir) else set()
    if ANTELOPEV2_EXPECTED_FILES.issubset(existing):
        ok("antelopev2 models already present, skipping download.")
        return True

    zip_path = os.path.join(dest_dir, "_antelopev2.zip")
    info(f"Downloading antelopev2 from InsightFace server …")
    info(f"URL: {ANTELOPEV2_URL}")

    try:
        urllib.request.urlretrieve(ANTELOPEV2_URL, zip_path, reporthook=_progress_hook)
        print()  # newline after progress bar
    except Exception as exc:
        print()
        err(f"Download failed: {exc}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

    # Verify zip is valid and extract
    info("Extracting antelopev2.zip …")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # The zip may contain a top-level folder; extract contents flat
            for member in zf.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                target = os.path.join(dest_dir, filename)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    except zipfile.BadZipFile as exc:
        err(f"Invalid zip file: {exc}")
        os.remove(zip_path)
        return False
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # Verify expected files are present
    extracted = set(os.listdir(dest_dir))
    missing = ANTELOPEV2_EXPECTED_FILES - extracted
    if missing:
        err(f"Missing files after extraction: {missing}")
        return False

    ok("antelopev2 extracted successfully.")
    return True


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

    # ── Step 2: antelopev2 ────────────────────────────────────────────────────
    section("Step 2 / 2  —  antelopev2 Face Encoder")
    antelopev2_dir = os.path.join("models", "antelopev2")
    success = download_antelopev2(antelopev2_dir)
    if not success:
        print(ANTELOPEV2_MANUAL_NOTE)

    # ── Done ──────────────────────────────────────────────────────────────────
    section("Done")
    print(
        f"\n  {GREEN}All automatic downloads complete.{RESET}\n"
        f"  The SDXL base model ({BOLD}wangqixun/YamerMIX_v8{RESET}, ~6 GB)\n"
        f"  will be downloaded automatically on first run of infer.py.\n"
    )


if __name__ == "__main__":
    main()
