#!/usr/bin/env python3
"""
setup_env.py
────────────
InstantID Face Transfer — 跨平台环境安装脚本

支持平台：
  Windows   NVIDIA GPU / CPU
  macOS     Apple Silicon (MPS) / Intel (CPU)
  Linux     NVIDIA GPU / CPU

NVIDIA CUDA 版本与 PyTorch wheel 的对应关系（自动检测）：
  CUDA >= 12.8  →  cu128  （Blackwell RTX 5000 系列必须）
  CUDA >= 12.4  →  cu124
  CUDA >= 12.1  →  cu121
  CUDA <  12.1  →  警告并退出（请升级驱动）

用法：
  python setup_env.py              # 自动检测平台与 CUDA 版本
  python setup_env.py --cu128      # 强制使用 CUDA 12.8 版 PyTorch（RTX 5000 系列）
  python setup_env.py --cu124      # 强制使用 CUDA 12.4 版 PyTorch
  python setup_env.py --cu121      # 强制使用 CUDA 12.1 版 PyTorch
  python setup_env.py --cpu        # 强制使用 CPU-only 版 PyTorch
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

ENV_NAME = "instantid"
PYTHON_VERSION = "3.10"

# ── ANSI 颜色（Windows Terminal / ANSICON 均支持）─────────────────────────────
if sys.platform == "win32":
    os.system("")          # 启用 Windows 控制台 ANSI 转义
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def sep():
    print(f"\n{BOLD}{CYAN}──────────────────────────────────────────────{RESET}")


def ok(msg: str):
    print(f"  {GREEN}✓{RESET}  {msg}")


def info(msg: str):
    print(f"  {YELLOW}→{RESET}  {msg}")


def err(msg: str):
    print(f"  {YELLOW}✗  {msg}{RESET}", file=sys.stderr)


def run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    info(" ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        err(f"Command failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    return result


# ── Conda helpers ─────────────────────────────────────────────────────────────

def find_conda() -> str:
    """Return path to conda executable, or exit with a helpful message."""
    exe = shutil.which("conda")
    if exe:
        return exe
    # Common Windows install paths
    if sys.platform == "win32":
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, "miniconda3", "Scripts", "conda.exe"),
            os.path.join(home, "anaconda3", "Scripts", "conda.exe"),
            r"C:\ProgramData\miniconda3\Scripts\conda.exe",
            r"C:\ProgramData\anaconda3\Scripts\conda.exe",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
    err(
        "conda not found.\n"
        "  Please install Miniconda: https://docs.conda.io/en/latest/miniconda.html\n"
        "  After installation, open a NEW terminal (or Anaconda Prompt on Windows)."
    )
    sys.exit(1)


def conda_env_exists(conda: str, name: str) -> bool:
    result = subprocess.run(
        [conda, "env", "list", "--json"],
        capture_output=True, text=True
    )
    try:
        data = json.loads(result.stdout)
        return any(
            os.path.basename(p) == name or p.endswith(f"/{name}") or p.endswith(f"\\{name}")
            for p in data.get("envs", [])
        )
    except json.JSONDecodeError:
        return False


# ── Platform / CUDA detection ─────────────────────────────────────────────────

def get_cuda_version() -> tuple[int, int] | None:
    """
    Parse the CUDA version from `nvidia-smi` output.
    Returns (major, minor) or None if nvidia-smi is unavailable.

    nvidia-smi prints a line like:
      | CUDA Version: 12.8     |
    or (older format):
      CUDA Version: 12.1
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        import re
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def cuda_version_to_backend(major: int, minor: int) -> str:
    """
    Map CUDA driver version to the PyTorch wheel tag.

    RTX 5000 (Blackwell, sm_120) requires CUDA 12.8+ and PyTorch cu128.
    RTX 4000 (Ada, sm_89) works with cu124+.
    RTX 3000/older work with cu121.
    """
    if (major, minor) >= (12, 8):
        return "nvidia_cu128"
    if (major, minor) >= (12, 4):
        return "nvidia_cu124"
    if (major, minor) >= (12, 1):
        return "nvidia_cu121"
    return None   # too old


def detect_backend() -> str:
    """
    Returns one of:
      'apple_silicon' | 'nvidia_cu128' | 'nvidia_cu124' | 'nvidia_cu121' | 'cpu'

    Detection order:
      1. Apple Silicon (Darwin arm64)
      2. nvidia-smi → parse CUDA version → select wheel
      3. Fallback to CPU
    """
    if platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        return "apple_silicon"

    cuda = get_cuda_version()
    if cuda is not None:
        backend = cuda_version_to_backend(*cuda)
        if backend is None:
            err(
                f"CUDA {cuda[0]}.{cuda[1]} is too old (minimum required: 12.1).\n"
                "  Please update your NVIDIA driver and CUDA toolkit, then retry."
            )
            sys.exit(1)
        return backend

    return "cpu"


# ── PyTorch install commands ──────────────────────────────────────────────────

TORCH_CMDS: dict = {
    "apple_silicon": [
        "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
    ],
    "nvidia_cu128": [
        "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ],
    "nvidia_cu124": [
        "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ],
    "nvidia_cu121": [
        "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ],
    "cpu": [
        "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ],
}

BACKEND_LABELS: dict = {
    "apple_silicon": "Apple Silicon (MPS)",
    "nvidia_cu128":  "NVIDIA CUDA 12.8 (Blackwell RTX 5000+)",
    "nvidia_cu124":  "NVIDIA CUDA 12.4",
    "nvidia_cu121":  "NVIDIA CUDA 12.1",
    "cpu":           "CPU-only",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="InstantID Face Transfer — Cross-platform environment setup"
    )
    parser.add_argument("--cu128", action="store_true", help="Force CUDA 12.8 PyTorch build (RTX 5000 Blackwell)")
    parser.add_argument("--cu124", action="store_true", help="Force CUDA 12.4 PyTorch build")
    parser.add_argument("--cu121", action="store_true", help="Force CUDA 12.1 PyTorch build")
    parser.add_argument("--cpu",   action="store_true", help="Force CPU-only PyTorch build")
    args = parser.parse_args()

    sep()
    print(f"  {BOLD}InstantID Face Transfer — Environment Setup{RESET}")
    print(f"  OS       : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.version.split()[0]}")
    sep()

    conda = find_conda()
    ok(f"conda found: {conda}")

    # ── Step 1: Create / reuse conda environment ──────────────────────────────
    sep()
    print(f"  {BOLD}Step 1 / 3  —  Conda environment{RESET}")
    sep()

    if conda_env_exists(conda, ENV_NAME):
        info(f"Environment '{ENV_NAME}' already exists, reusing it.")
    else:
        info(f"Creating conda environment: {ENV_NAME} (Python {PYTHON_VERSION}) …")
        run([conda, "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"])
        ok("Environment created.")

    # ── Step 2: Install PyTorch ───────────────────────────────────────────────
    sep()
    print(f"  {BOLD}Step 2 / 3  —  PyTorch installation{RESET}")
    sep()

    if args.cu128:
        backend = "nvidia_cu128"
    elif args.cu124:
        backend = "nvidia_cu124"
    elif args.cu121:
        backend = "nvidia_cu121"
    elif args.cpu:
        backend = "cpu"
    else:
        backend = detect_backend()

    cuda = get_cuda_version()
    if cuda and backend.startswith("nvidia"):
        info(f"nvidia-smi reports CUDA {cuda[0]}.{cuda[1]}")
        # Warn if user forces a wheel that requires higher CUDA than available
        required = {"nvidia_cu128": (12, 8), "nvidia_cu124": (12, 4), "nvidia_cu121": (12, 1)}
        req = required.get(backend)
        if req and cuda < req:
            err(
                f"You requested {BACKEND_LABELS[backend]} but your driver only supports "
                f"CUDA {cuda[0]}.{cuda[1]}.\n"
                f"  Please update your NVIDIA driver to support CUDA {req[0]}.{req[1]}+."
            )
            sys.exit(1)

    info(f"Backend detected: {BACKEND_LABELS[backend]}")
    run([conda, "run", "--no-capture-output", "-n", ENV_NAME] + TORCH_CMDS[backend])
    ok(f"PyTorch ({BACKEND_LABELS[backend]}) installed.")

    # ── Step 3: Install remaining dependencies ────────────────────────────────
    sep()
    print(f"  {BOLD}Step 3 / 3  —  Python dependencies{RESET}")
    sep()

    run([
        conda, "run", "--no-capture-output", "-n", ENV_NAME,
        "pip", "install", "-r", "requirements.txt",
    ])
    ok("All dependencies installed.")

    # ── Done ──────────────────────────────────────────────────────────────────
    sep()
    print(f"\n  {GREEN}{BOLD}Setup complete!{RESET}\n")
    activate_cmd = (
        f"conda activate {ENV_NAME}"
        if sys.platform != "win32"
        else f"conda activate {ENV_NAME}   (在 Anaconda Prompt / PowerShell 中执行)"
    )
    print("  Next steps:\n")
    print(f"    1.  {activate_cmd}")
    print("    2.  python download_models.py      # 下载 InstantID 权重")
    print("        (按提示手动下载 antelopev2)\n")
    print("    3.  python infer.py \\")
    print("            --source source/<your_face.jpg> \\")
    print("            --target target/<sketch.jpg>")
    print("")
    sep()


if __name__ == "__main__":
    main()
