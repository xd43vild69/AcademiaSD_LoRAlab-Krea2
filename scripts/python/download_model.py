"""Descargar modelo Krea-2-NF4 desde HuggingFace.

Se ejecuta en el entrypoint del Docker si el modelo no existe.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download


def download_krea2_model():
    """Descargar modelo Krea-2-NF4 desde HF."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
    model_path = Path(os.getenv("HF_HOME", "/data/models")) / "Krea-2-NF4"

    # Verificar si ya existe
    if model_path.exists() and (model_path / "model_index.json").exists():
        print(f"[OK] Modelo ya existe en {model_path}")
        return True

    # Crear directorio
    model_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Descargando Krea-2-NF4 desde HuggingFace a {model_path}...")
    try:
        snapshot_download(
            repo_id="multimodalart/Krea-2-NF4",  # ID del modelo en HF
            local_dir=str(model_path),
            token=hf_token if hf_token else None,
            resume_download=True,
            allow_patterns=["*.safetensors", "*.json", "*.py"],
        )
        print("[OK] Modelo descargado exitosamente")
        return True
    except Exception as e:
        print(f"[!] Error al descargar modelo: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = download_krea2_model()
    sys.exit(0 if success else 1)
