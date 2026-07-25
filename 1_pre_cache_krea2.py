# -*- coding: utf-8 -*-
"""
1_pre_cache_krea2.py — Pre-caché de latentes + embeddings para Krea 2 (RAW)
Pre-cache of latents + embeddings for Krea 2 (RAW)

Lee configuración desde pre_cache_settings.json si existe.
Reads configuration from pre_cache_settings.json if present.
"""
import os
import gc
import math
import json
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import DiffusionPipeline
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Krea-2-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./cached_data_krea2",
    "target_area": 512 * 512,
    "max_side": 1280,
    "multiple": 16,
    "max_seq_len": 128,
    "project_name": "",
    "trigger_word": "",
    "resolutions": [],
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = "pre_cache_settings.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} not found, using defaults / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID     = cfg.get("model_id",     DEFAULTS["model_id"])
DATASET_PATH = cfg.get("dataset_path", DEFAULTS["dataset_path"])
TARGET_AREA  = cfg.get("target_area",  DEFAULTS["target_area"])
MAX_SIDE     = cfg.get("max_side",     DEFAULTS["max_side"])
MULTIPLE     = cfg.get("multiple",     DEFAULTS["multiple"])
MAX_SEQ_LEN  = cfg.get("max_seq_len",  DEFAULTS["max_seq_len"])
TRIGGER_WORD = cfg.get("trigger_word", "")
PROJECT_NAME = cfg.get("project_name", "").strip()

# Resolución progresiva: si 'resolutions' trae una lista de áreas, se pre-cachea
# el dataset a cada una en un subdirectorio {label}/. Si está vacía, se usa la
# resolución única de 'target_area' (comportamiento clásico, retrocompatible).
RESOLUTIONS = cfg.get("resolutions", DEFAULTS["resolutions"]) or []


def res_label(area):
    """Etiqueta corta del subdir a partir del área (ej. 589824 -> '768')."""
    return str(int(round(math.sqrt(area))))

# Formato automático de carpeta de caché según nombre del proyecto
if PROJECT_NAME:
    CACHE_DIR = f"./cached_data_krea2_{PROJECT_NAME}"
else:
    CACHE_DIR = cfg.get("cache_dir", DEFAULTS["cache_dir"])

# Validar que Múltiplo sea únicamente 16, 32 o 64 (Default: 16).
# 8 no vale: pack_latents() del entrenador divide las dimensiones latentes (px/8)
# entre 2, así que los píxeles deben ser múltiplo de 16 o el empaquetado se
# corrompe en silencio.
if MULTIPLE not in (16, 32, 64):
    print(f"⚠ Invalid Multiple {MULTIPLE}. Defaulting to 16 / Múltiplo inválido {MULTIPLE}. Usando 16 por defecto.")
    MULTIPLE = 16

print(f"  Model ID / ID Modelo        : {MODEL_ID}")
print(f"  Project Name / Proyecto     : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra      : {TRIGGER_WORD}")
print(f"  Dataset Path / Ruta Dataset : {DATASET_PATH}")
print(f"  Cache Dir / Carpeta Caché   : {CACHE_DIR}")
if RESOLUTIONS:
    print(f"  Resolutions / Progresiva    : {[res_label(a)+'²' for a in RESOLUTIONS]}")
else:
    print(f"  Target Area / Área Objetivo : {TARGET_AREA} px²")
print(f"  Max Side / Lado Máximo      : {MAX_SIDE}")
print(f"  Multiple / Múltiplo         : {MULTIPLE}")
print(f"  Max Seq Len / Long. Sec.    : {MAX_SEQ_LEN}")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def free_vram(*tensors):
    for t in tensors:
        del t
    gc.collect()
    torch.cuda.empty_cache()


def bucket_size(w: int, h: int, area=None):
    """(ancho, alto) múltiplos de MULTIPLE, área ≈ `area`, conservando el aspecto."""
    if area is None:
        area = TARGET_AREA
    ar = w / h
    bh = math.sqrt(area / ar)
    bw = ar * bh
    bw = max(MULTIPLE, round(bw / MULTIPLE) * MULTIPLE)
    bh = max(MULTIPLE, round(bh / MULTIPLE) * MULTIPLE)
    if max(bw, bh) > MAX_SIDE:
        s = MAX_SIDE / max(bw, bh)
        bw = max(MULTIPLE, int(bw * s) // MULTIPLE * MULTIPLE)
        bh = max(MULTIPLE, int(bh * s) // MULTIPLE * MULTIPLE)
    return bw, bh


def ensure_model_downloaded(local_path, repo_id):
    if os.path.exists(local_path) and os.path.isdir(local_path):
        has_content = any(
            os.path.exists(os.path.join(local_path, f))
            for f in ["index.json", "model_index.json", "config.json"]
        ) or len(os.listdir(local_path)) > 0
        if has_content:
            print(f"✓ Local model found at / Modelo local encontrado en: {local_path}")
            return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")
    print(f"  This may take several minutes / Esto puede tardar varios minutos...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with / Se requiere 'huggingface_hub'. Instálalo con:\n"
            "  pip install huggingface_hub"
        )

    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")
    dl_kwargs = {
        "repo_id": repo_id,
        "local_dir": local_path,
    }
    if hf_token:
        dl_kwargs["token"] = hf_token

    downloaded_path = snapshot_download(**dl_kwargs)
    print(f"✓ Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def preprocess_krea2():
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {DATASET_PATH}")
        return

    # Auto-limpieza de archivos fantasma ocultos de macOS (._* y .DS_Store)
    for f in os.listdir(DATASET_PATH):
        if f.startswith("._") or f == ".DS_Store":
            try:
                os.remove(os.path.join(DATASET_PATH, f))
            except Exception:
                pass

    archivos_img = sorted(f for f in os.listdir(DATASET_PATH)
                          if not f.startswith(".") and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not archivos_img:
        print(f"[!] No images found in '{DATASET_PATH}'. Please add images and optional .txt files. / No hay imágenes en '{DATASET_PATH}'. Añade imágenes y sus .txt opcionales.")
        return

    ensure_model_downloaded(
        local_path=MODEL_ID,
        repo_id="AcademiaSD/Krea-2-NF4-for-LoRA-Training"
    )

    print("Loading VAE (Qwen-Image) and Text Encoder (Qwen3-VL-4B)... / Cargando VAE y Text Encoder de Krea-2...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        transformer=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    if not hasattr(pipe.vae.config, "latents_mean") or not hasattr(pipe.vae.config, "latents_std"):
        raise RuntimeError("This VAE lacks latents_mean/latents_std configuration / Este VAE no tiene latents_mean/latents_std")

    z_dim = pipe.vae.config.z_dim
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda", dtype=torch.float32).view(1, z_dim, 1, 1, 1)
    latents_std  = torch.tensor(pipe.vae.config.latents_std,  device="cuda", dtype=torch.float32).view(1, z_dim, 1, 1, 1)
    print(f"VAE z_dim={z_dim} | Channel normalization OK / Normalización por canal OK")

    # Objetivos de resolución: cada uno es una caché autocontenida. En modo
    # progresivo van a subdirs {label}/; en modo clásico, a CACHE_DIR directamente.
    if RESOLUTIONS:
        targets = [(os.path.join(CACHE_DIR, res_label(a)), a) for a in RESOLUTIONS]
    else:
        targets = [(CACHE_DIR, TARGET_AREA)]
    for out_dir, _ in targets:
        os.makedirs(out_dir, exist_ok=True)

    def vae_latent(img_pil, area):
        """Redimensiona/recorta al bucket de `area` y devuelve el latente normalizado."""
        bw, bh = bucket_size(img_pil.width, img_pil.height, area)
        ratio = max(bw / img_pil.width, bh / img_pil.height)
        im = img_pil.resize((math.ceil(img_pil.width * ratio), math.ceil(img_pil.height * ratio)), Image.LANCZOS)
        left = (im.width - bw) // 2
        top  = (im.height - bh) // 2
        im   = im.crop((left, top, left + bw, top + bh))

        img_tensor = F_vision.pil_to_tensor(im).unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W]
        img_tensor = (img_tensor.float() / 127.5) - 1.0
        img_tensor = img_tensor.to("cuda", dtype=torch.bfloat16)

        z = pipe.vae.encode(img_tensor).latent_dist.sample().float()
        latent = (z - latents_mean) / latents_std
        latent = latent[:, :, 0].to(torch.bfloat16).cpu()
        free_vram(img_tensor, z)
        return bw, bh, latent

    with torch.inference_mode():
        # El embed negativo es independiente de la resolución: se guarda en cada caché.
        neg_embed, neg_mask = pipe.encode_prompt(prompt="", max_sequence_length=MAX_SEQ_LEN)
        for out_dir, _ in targets:
            torch.save(neg_embed.cpu(), os.path.join(out_dir, "_neg_embed.pt"))
            torch.save(neg_mask.cpu(),  os.path.join(out_dir, "_neg_mask.pt"))
        del neg_embed, neg_mask

        for idx, archivo in enumerate(archivos_img, 1):
            nombre_base = os.path.splitext(archivo)[0]
            ruta_texto  = os.path.join(DATASET_PATH, f"{nombre_base}.txt")
            print(f"[{idx}/{len(archivos_img)}] Processing / Procesando: {archivo}")

            try:
                img = Image.open(os.path.join(DATASET_PATH, archivo)).convert("RGB")
            except Exception as err:
                print(f"[!] Warning: Cannot open image / No se pudo abrir la imagen '{archivo}': {err}")
                continue

            # ── 1. TEXTO → EMBEDDINGS (una sola vez; es independiente de la resolución) ──
            prompt = ""
            if os.path.exists(ruta_texto):
                with open(ruta_texto, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()

            if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
                prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")

            embeds, mask = pipe.encode_prompt(prompt=prompt, max_sequence_length=MAX_SEQ_LEN)
            embeds_cpu, mask_cpu = embeds.cpu(), mask.cpu()
            del embeds, mask

            # ── 2. IMAGEN → LATENTES (VAE), una vez por resolución ──────────
            sizes = []
            for out_dir, area in targets:
                bw, bh, latent = vae_latent(img, area)
                torch.save(latent, os.path.join(out_dir, f"{nombre_base}_latent.pt"))
                torch.save(embeds_cpu, os.path.join(out_dir, f"{nombre_base}_embed.pt"))
                torch.save(mask_cpu,   os.path.join(out_dir, f"{nombre_base}_mask.pt"))
                sizes.append(f"{bw}x{bh}")
                free_vram(latent)

            print(f"   [OK] {' · '.join(sizes)} | Latents & Embeddings cached / Latentes y embeddings cacheados.")

    free_vram(pipe)
    print("\n✓ Pre-caching finished! VRAM freed / ¡Pre-caché finalizado! VRAM liberada.")


if __name__ == "__main__":
    preprocess_krea2()