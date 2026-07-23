"""Download the reference GGUF model for calibration.

Downloads Qwen2.5-1.5B-Instruct Q4_K_M (~1.12 GB) from HuggingFace into
the models/ directory. Verifies file size after download.

Usage:
    python scripts/fetch_reference_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
EXPECTED_SIZE_MB = 1120  # ~1.12 GB


def main() -> None:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    target = models_dir / FILENAME
    if target.exists():
        size_mb = target.stat().st_size / (1024 * 1024)
        if abs(size_mb - EXPECTED_SIZE_MB) < 50:
            print(f"Model already present: {target} ({size_mb:.0f} MB)")
            return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {FILENAME} from {REPO_ID}...")
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=str(models_dir))

    size_mb = target.stat().st_size / (1024 * 1024)
    if abs(size_mb - EXPECTED_SIZE_MB) > 100:
        print(f"WARNING: unexpected file size {size_mb:.0f} MB (expected ~{EXPECTED_SIZE_MB} MB)")
        sys.exit(1)

    print(f"Downloaded: {target} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
