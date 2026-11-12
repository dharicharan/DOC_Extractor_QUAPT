"""
One-stop Kaggle launcher. The notebook only clones/pulls the repo and runs this file, so every
change pushed to git reaches Kaggle without editing the notebook.

Needs: Internet ON and a GPU in the Kaggle session options.
Run:   %run /kaggle/working/DOC_Extractor_QUAPT/kaggle_launch.py
"""

import importlib.util
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(REPO, "src", "doc_extractor")

# (model id, required). Weights are only downloaded here; app.py loads them onto the GPU itself.
MODELS = [
    ("Qwen/Qwen3-VL-8B-Instruct", True),       # router + default extractor
    ("zai-org/GLM-OCR", False),                 # experimental OCR engine
    ("Qwen/Qwen3-4B-Instruct-2507", False),     # default text step after GLM-OCR (pure text, non-thinking)
    ("Qwen/Qwen2.5-7B-Instruct", False),        # comparison text step
]


def run(cmd: str) -> int:
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True).returncode


def ensure_dependencies() -> None:
    missing = [m for m in ("gradio", "qwen_vl_utils", "bitsandbytes", "transformers", "accelerate")
               if importlib.util.find_spec(m) is None]
    if missing or importlib.util.find_spec("doc_extractor") is None:
        print("Installing dependencies (first run only)...")
        run("pip install -q uv")
        run(f"uv pip install --system -r {REPO}/requirements.txt")
        run(f"uv pip install --system -e {REPO}")
        run("uv pip install --system qwen-vl-utils transformers accelerate bitsandbytes gradio")
    else:
        print("Dependencies already installed.")


def download_models() -> None:
    from huggingface_hub import snapshot_download

    for model_id, required in MODELS:
        try:
            print(f"Downloading {model_id} ...")
            snapshot_download(model_id)
            print("  cached.")
        except Exception as e:  # noqa: BLE001
            msg = (f"Download failed for {model_id}: {type(e).__name__}. "
                   "Turn Internet ON in the Kaggle session options and select a GPU.")
            if required:
                raise SystemExit(msg)
            print("  skipped (optional):", msg)


def main() -> None:
    print(subprocess.run(["git", "-C", REPO, "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
    ensure_dependencies()
    download_models()
    run("fuser -k 7860/tcp 2>/dev/null || true")
    os.chdir(APP_DIR)
    subprocess.run([sys.executable, "app.py"])


if __name__ == "__main__":
    main()
