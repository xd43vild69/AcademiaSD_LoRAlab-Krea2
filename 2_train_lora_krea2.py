# -*- coding: utf-8 -*-
"""
2_train_lora_krea2.py — Entrenamiento LoRA para Krea 2 (RAW) con NF4 / LoRA Training for Krea 2

Lee configuración desde train_settings.json si existe.
Reads configuration from train_settings.json if present.
"""
import os
import gc
import math
import time
import random
import json
import signal
import sys
from collections import defaultdict

# Debe fijarse antes de que se inicialice el asignador CUDA. Reduce la
# fragmentación de VRAM, que es el modo de fallo típico en GPUs de 12 GB al
# entrenar a 768x768 o más.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import bitsandbytes as bnb
from safetensors.torch import save_file, load
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open
from bitsandbytes.functional import QuantState

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Carpetas contenedoras: todo lo generado se agrupa aquí en vez de en la raíz.
CACHE_ROOT  = "./cached_data_local"
OUTPUT_ROOT = "./output_local"

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Krea-2-NF4",
    "cache_dir": f"{CACHE_ROOT}/default",
    "output_dir": f"{OUTPUT_ROOT}/default",
    "total_steps": 1200,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lora_rank": 16,
    "lora_alpha": 32,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 25,
    "seed": 42,
    "timestep_sampling": "krea2_shift",
    "preview_every": 0,
    "preview_steps": 28,
    "preview_cfg": 3.5,
    "preview_caption_mode": "first",
    "project_name": "",
    "trigger_word": "",
    "lora_target": "all",
    "compact_text": True,
    "init_lora_from": "",
    "gradient_checkpointing": True,
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
# El orquestador de resolución progresiva pasa un fichero por fase vía esta
# env-var para no pisar el train_settings.json del usuario.
CONFIG_PATH = os.environ.get("TRAIN_SETTINGS_PATH", "train_settings.json")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[OK] Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"[!] {CONFIG_PATH} not found, using default values / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID          = cfg.get("model_id",          DEFAULTS["model_id"])
TOTAL_STEPS       = cfg.get("total_steps",       DEFAULTS["total_steps"])
BATCH_SIZE        = cfg.get("batch_size",        DEFAULTS["batch_size"])
GRAD_ACCUM_STEPS  = cfg.get("grad_accum_steps",  DEFAULTS["grad_accum_steps"])
LR                = cfg.get("lr",                DEFAULTS["lr"])
MIN_LR_RATIO      = cfg.get("min_lr_ratio",      DEFAULTS["min_lr_ratio"])
WARMUP_STEPS      = cfg.get("warmup_steps",      DEFAULTS["warmup_steps"])
LORA_RANK         = cfg.get("lora_rank",         DEFAULTS["lora_rank"])
LORA_ALPHA        = cfg.get("lora_alpha",        DEFAULTS["lora_alpha"])
WEIGHT_DECAY      = cfg.get("weight_decay",      DEFAULTS["weight_decay"])
MAX_GRAD_NORM     = cfg.get("max_grad_norm",     DEFAULTS["max_grad_norm"])
SAVE_EVERY        = cfg.get("save_every",        DEFAULTS["save_every"])
SEED              = cfg.get("seed",              DEFAULTS["seed"])
TIMESTEP_SAMPLING = cfg.get("timestep_sampling", DEFAULTS["timestep_sampling"])
PREVIEW_EVERY     = cfg.get("preview_every",     DEFAULTS["preview_every"])
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
PREVIEW_CAPTION_MODE = cfg.get("preview_caption_mode", DEFAULTS["preview_caption_mode"])
TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()
LORA_TARGET       = str(cfg.get("lora_target", DEFAULTS["lora_target"])).strip().lower()
COMPACT_TEXT      = bool(cfg.get("compact_text", DEFAULTS["compact_text"]))
INIT_LORA_FROM    = str(cfg.get("init_lora_from", DEFAULTS["init_lora_from"])).strip()
GRAD_CHECKPOINTING = bool(cfg.get("gradient_checkpointing", DEFAULTS["gradient_checkpointing"]))

if LORA_TARGET not in ("all", "attn", "attn+ff"):
    print(f"[!] Invalid lora_target '{LORA_TARGET}'. Using 'all' / lora_target inválido. Usando 'all'.")
    LORA_TARGET = "all"

# La compactación de texto elimina los tokens de relleno de cada caption, lo que
# permite prescindir de la máscara de atención. Con máscara + GQA, PyTorch no
# puede usar ni flash ni mem-efficient y cae al backend `math`, que materializa
# la matriz [B, heads, S, S] completa (~568 MB a 768x768). Sin máscara usa flash.
# Sólo es aplicable con batch 1: al compactar, cada muestra queda con una
# longitud de texto distinta y torch.cat dejaría de funcionar.
if COMPACT_TEXT and BATCH_SIZE > 1:
    print("[!] compact_text requires batch_size 1; disabling / compact_text requiere batch_size 1; desactivado.")
    COMPACT_TEXT = False

# Formato automático de carpetas según el nombre del proyecto.
# Sin project_name se respetan cache_dir/output_dir explícitos: así es como
# run_progressive.py apunta cada fase a su subdir de resolución y a phaseN_*.
if PROJECT_NAME:
    CACHE_DIR  = f"{CACHE_ROOT}/{PROJECT_NAME}"
    OUTPUT_DIR = f"{OUTPUT_ROOT}/{PROJECT_NAME}"
else:
    CACHE_DIR  = cfg.get("cache_dir",  DEFAULTS["cache_dir"])
    OUTPUT_DIR = cfg.get("output_dir", DEFAULTS["output_dir"])

print(f"  Model ID / ID Modelo     : {MODEL_ID}")
print(f"  Project / Proyecto       : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra   : {TRIGGER_WORD}")
print(f"  Cache Dir / Carpeta Caché: {CACHE_DIR}")
print(f"  Output Dir / Salida      : {OUTPUT_DIR}")
print(f"  Total Steps / Pasos      : {TOTAL_STEPS}")
print(f"  Learning Rate / LR       : {LR}")
print(f"  LoRA Rank/Alpha          : {LORA_RANK}/{LORA_ALPHA}")
print(f"  LoRA Target / Capas      : {LORA_TARGET}")
print(f"  Batch / Grad Accum       : {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  Compact Text / Compactar : {'ON' if COMPACT_TEXT else 'OFF'}")
print(f"  Grad Checkpointing       : {'ON' if GRAD_CHECKPOINTING else 'OFF'}")
if INIT_LORA_FROM:
    print(f"  Init LoRA from / Fase prev: {INIT_LORA_FROM}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE   = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE  = os.path.join(OUTPUT_DIR, "current_step.txt")

torch.manual_seed(SEED)
random.seed(SEED)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def patch_attention_for_low_vram():
    """Evita el backend `math` de SDPA cuando hay que conservar la máscara.

    PyTorch no admite `attn_mask` junto con `enable_gqa=True` ni en flash ni en
    mem-efficient, así que recae en `math`, que materializa la matriz completa
    [B, heads, S, S]. Expandiendo K/V al número de cabezas de Q se puede pasar
    `enable_gqa=False` y mem-efficient vuelve a estar disponible: el coste son
    unas decenas de MB de K/V frente a cientos de MB de scores.

    Sin máscara (ver COMPACT_TEXT) flash ya funciona con GQA y esto no actúa.
    """
    from diffusers.models.transformers import transformer_krea2

    original = transformer_krea2.dispatch_attention_fn

    def dispatch(query, key, value, *args, attn_mask=None, enable_gqa=False, **kwargs):
        if attn_mask is not None and enable_gqa and key.shape[2] != query.shape[2]:
            repeats = query.shape[2] // key.shape[2]
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)
            enable_gqa = False
        return original(query, key, value, *args, attn_mask=attn_mask, enable_gqa=enable_gqa, **kwargs)

    transformer_krea2.dispatch_attention_fn = dispatch
    print("[OK] Attention patched to avoid the SDPA math backend / Atención parcheada para evitar el backend math.")


def ensure_model_downloaded(local_path, repo_id):
    if os.path.exists(local_path) and os.path.isdir(local_path):
        has_content = any(
            os.path.exists(os.path.join(local_path, f))
            for f in ["index.json", "model_index.json", "config.json"]
        ) or len(os.listdir(local_path)) > 0
        if has_content:
            print(f"[OK] Local model found at / Modelo local encontrado en: {local_path}")
            return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")
    print(f"  This may take several minutes / Esto puede tardar varios minutos...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with / Se requiere 'huggingface_hub':\n"
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

    print(f"[OK] Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=6400,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def sample_sigma(batch_size, image_seq_len, device, shift_cfg):
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
    mu = calculate_shift(image_seq_len, *shift_cfg)
    e_mu = math.exp(mu)
    sigma = e_mu / (e_mu + (1.0 / u.clamp(1e-6, 1 - 1e-6) - 1.0))
    return sigma.clamp(1e-4, 1.0 - 1e-4)


def pack_latents(x):
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // 2) * (W // 2), C * 4)


def unpack_latents(x, H, W):
    B, _, C = x.shape
    x = x.view(B, H // 2, W // 2, C // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C // 4, H, W)


def prepare_position_ids(text_seq_len, grid_h, grid_w, device):
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_h, grid_w, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
    return torch.cat([text_ids, image_ids.reshape(grid_h * grid_w, 3)], dim=0)


SKIP_QUANT = ("img_in", "time_embed", "time_mod_proj", "txt_in", "final_layer")


def quantize_to_nf4_(module, prefix=""):
    from bitsandbytes.nn import Linear4bit, Params4bit
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(s in full for s in SKIP_QUANT):
            w = child.weight.data.float().contiguous()
            new_layer = Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None, quant_type="nf4",
                compute_dtype=torch.bfloat16,
            )
            new_layer.weight = Params4bit(w, requires_grad=False, quant_type="nf4")
            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, new_layer)
            del child, w
        else:
            quantize_to_nf4_(child, full)


def load_nf4_cache_(transformer, cache_dir):
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json not found in NF4 cache / No existe index.json en caché NF4: {cache_dir}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    quantized = index.get("quantized", {})
    weights_dir = os.path.join(cache_dir, "weights")
    replaced = 0

    def get_parent_module(root, module_name):
        parts = module_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    for name, info in quantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NF4 weight file not found / No existe archivo NF4: {filepath}")

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")
            bias_data = None
            if info.get("bias", False):
                bias_data = f.get_tensor("bias")

            qs_dict = {}
            for key in f.keys():
                if not key.startswith("quant_state."):
                    continue
                qs_key = key[len("quant_state."):]
                qs_dict[qs_key] = f.get_tensor(key)

            packed_qs = {
                "absmax": qs_dict["absmax"],
                "nested_absmax": qs_dict["nested_absmax"],
                "nested_quant_map": qs_dict["nested_quant_map"],
                "quant_map": qs_dict["quant_map"],
                "quant_state.bitsandbytes__nf4": qs_dict["quant_state.bitsandbytes__nf4"],
            }

            quant_state = QuantState.from_dict(packed_qs, device="cpu")

        new_weight = Params4bit(
            weight_data,
            requires_grad=False,
            quant_type="nf4",
            quant_storage=torch.uint8,
        )
        new_weight.quant_state = quant_state
        new_weight.bnb_quantized = True

        new_layer = Linear4bit(
            info["in_features"],
            info["out_features"],
            bias=info["bias"],
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )
        new_layer.weight = new_weight
        if bias_data is not None:
            new_layer.bias = torch.nn.Parameter(bias_data, requires_grad=False)

        setattr(parent, child_name, new_layer)
        replaced += 1

    print(f"Reconstructed NF4 layers / Capas NF4 reconstruidas: {replaced}")

    verified = 0
    for name, layer in transformer.named_modules():
        if isinstance(layer, Linear4bit):
            if getattr(layer.weight, "bnb_quantized", False):
                if layer.weight.quant_state is not None:
                    verified += 1

    print(f"Verified NF4 layers / Capas NF4 verificadas: {verified}")

    if verified != replaced:
        raise RuntimeError("NF4 Verification mismatch / La verificación NF4 no coincide")

    print("[OK] NF4 cache loaded successfully / Caché NF4 cargada correctamente.")
    return transformer


def _export_lora(model, path):
    clean = {}
    for k, v in model.state_dict().items():
        if "lora_" not in k:
            continue
        new_key = "transformer." + k.replace("base_model.model.", "")
        clean[new_key] = v.to(torch.bfloat16).cpu().contiguous()
    save_file(clean, path, metadata={"format": "pt"})


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            from diffusers import AutoencoderKLQwenImage
            cls.vae = AutoencoderKLQwenImage.from_pretrained(
                MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
        return cls.vae


def run_preview(model, scheduler, embed, mask, neg, size, step, shift_cfg):
    H, W = size
    gh, gw = H // 16, W // 16
    device = "cuda"
    was_training = model.training
    model.eval()

    g = torch.Generator(device=device).manual_seed(SEED)
    latents = torch.randn((1, 16, H // 8, W // 8), generator=g, device=device, dtype=torch.bfloat16)
    latents = pack_latents(latents)
    pos_ids = prepare_position_ids(embed.shape[1], gh, gw, device)
    embed = embed.to(device)
    mask = mask.to(device) if mask is not None else None
    neg_pos_ids = None
    if neg is not None:
        neg = (neg[0].to(device), neg[1].to(device) if neg[1] is not None else None)
        # Compactado, el prompt negativo tiene menos tokens que el positivo, así
        # que necesita sus propios position_ids.
        neg_pos_ids = (pos_ids if neg[0].shape[1] == embed.shape[1]
                       else prepare_position_ids(neg[0].shape[1], gh, gw, device))

    sigmas = np.linspace(1.0, 1.0 / PREVIEW_STEPS, PREVIEW_STEPS)
    mu = calculate_shift(latents.shape[1], *shift_cfg)
    scheduler.set_timesteps(PREVIEW_STEPS, device=device, sigmas=sigmas, mu=mu)

    with torch.no_grad():
        for t in scheduler.timesteps:
            tt = (t / scheduler.config.num_train_timesteps).expand(1).to(torch.bfloat16)
            pred = model(hidden_states=latents, encoder_hidden_states=embed, timestep=tt,
                         position_ids=pos_ids, encoder_attention_mask=mask, return_dict=False)[0]
            if neg is not None:
                pred_u = model(hidden_states=latents, encoder_hidden_states=neg[0], timestep=tt,
                               position_ids=neg_pos_ids, encoder_attention_mask=neg[1], return_dict=False)[0]
                pred = pred + PREVIEW_CFG * (pred - pred_u)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]

        vae = VaeHolder.get().to(device)
        lat = unpack_latents(latents, H // 8, W // 8).to(vae.dtype).unsqueeze(2)
        mean = torch.tensor(vae.config.latents_mean, device=device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
        std  = torch.tensor(vae.config.latents_std,  device=device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
        img = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
        img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        vae.to("cpu")

    from PIL import Image
    out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
    Image.fromarray(img).save(out)
    print(f"\n  ↳ Preview saved to / Preview guardada: {out}")
    if was_training:
        model.train()
    free_vram()


# ── ENTRENAMIENTO / TRAINING ─────────────────────────────────────────────────
def train_krea2():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    patch_attention_for_low_vram()

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print(f"\n[!] ERROR: Cache directory '{CACHE_DIR}' is empty or does not exist.")
        print(f"[!] Please run Pre-Cache first! / ¡Por favor ejecuta el Pre-Caché primero!")
        return

    ensure_model_downloaded(
        local_path=MODEL_ID,
        repo_id="AcademiaSD/Krea-2-NF4-for-LoRA-Training"
    )

    print("Loading Krea-2 Transformer... / Cargando Transformer de Krea-2...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        vae=None,
        text_encoder=None,
        torch_dtype=torch.bfloat16,
    )

    transformer = pipe.transformer
    scheduler   = pipe.scheduler

    del pipe
    free_vram()

    shift_cfg = (
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 6400),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )

    NF4_CACHE_DIR = MODEL_ID

    if os.path.exists(os.path.join(NF4_CACHE_DIR, "index.json")):
        print("\n¡NF4 CACHE DETECTED! / ¡CACHÉ NF4 DETECTADA!")
        print("Skipping NF4 quantization... / No se ejecutará cuantización NF4.")
        t0 = time.time()
        transformer = load_nf4_cache_(transformer, NF4_CACHE_DIR)
        print(f"[NF4] Cache loaded in / Caché cargada en {time.time() - t0:.1f}s", flush=True)
        transformer.to("cuda")
        free_vram()
        print(f"Transformer 12B pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    else:
        print("\nNo NF4 cache found. Performing quantization... / No se encontró caché NF4. Ejecutando cuantización...")
        quantize_to_nf4_(transformer)
        transformer.to("cuda")
        free_vram()
        print(f"Transformer 12B pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    if GRAD_CHECKPOINTING:
        transformer.enable_gradient_checkpointing()
    else:
        # Sin recompute: más rápido pero más VRAM de activaciones. Sólo apto en las
        # fases de baja resolución donde la VRAM sobra (lo decide el orquestador).
        print("Gradient checkpointing OFF (faster, more VRAM) / Checkpointing desactivado.")

    all_linears = [name for name, m in transformer.named_modules()
                   if isinstance(m, (torch.nn.Linear, bnb.nn.Linear4bit))]

    def keep(name):
        # 'all' incluye también los bloques de text_fusion; los presets reducidos
        # se limitan a los transformer_blocks de imagen, que es donde el LoRA rinde.
        if LORA_TARGET == "all":
            return True
        if not name.startswith("transformer_blocks."):
            return False
        if ".attn." in name:
            return True
        return LORA_TARGET == "attn+ff" and ".ff." in name

    target_modules = [n for n in all_linears if keep(n)]
    if not target_modules:
        print(f"[!] lora_target '{LORA_TARGET}' matched no layers; falling back to 'all' / no coincidió con ninguna capa.")
        target_modules = all_linears
    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)}/{len(all_linears)} ({LORA_TARGET})")

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=target_modules, use_dora=False, init_lora_weights=True,
    )

    model = get_peft_model(transformer, lora_config)

    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter in module.lora_A.values():
                adapter.to(dtype=torch.bfloat16)
        if hasattr(module, "lora_B"):
            for adapter in module.lora_B.values():
                adapter.to(dtype=torch.bfloat16)
        if hasattr(module, "lora_embedding_A"):
            for adapter in module.lora_embedding_A.values():
                adapter.data = adapter.data.to(torch.bfloat16)
        if hasattr(module, "lora_embedding_B"):
            for adapter in module.lora_embedding_B.values():
                adapter.data = adapter.data.to(torch.bfloat16)

    model.print_trainable_parameters()

    def _make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    transformer.img_in.register_forward_hook(_make_inputs_require_grad)

    trainable = [p for p in model.parameters() if p.requires_grad]
    # Paged: vuelca el estado del optimizador a RAM bajo presión de VRAM en lugar
    # de reventar con OOM en los picos de las resoluciones altas.
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(step):
        # El LR sólo se aplica en actualizaciones reales del optimizador (1 de cada
        # GRAD_ACCUM_STEPS pasos de bucle), así que el schedule se mide en updates,
        # no en pasos de bucle. WARMUP_STEPS se interpreta como updates del optimizador.
        update = step / max(1, GRAD_ACCUM_STEPS)
        total_updates = max(1, TOTAL_STEPS / max(1, GRAD_ACCUM_STEPS))
        if update < WARMUP_STEPS:
            return LR * update / max(1, WARMUP_STEPS)
        prog = (update - WARMUP_STEPS) / max(1e-9, total_updates - WARMUP_STEPS)
        return LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    # ── RESTAURACIÓN EXACTA DE CHECKPOINT / CHECKPOINT RESUME ─────────────────
    start_step = 0
    lora_weights_path = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    if os.path.exists(STEP_FILE) and os.path.exists(OPT_FILE) and os.path.exists(lora_weights_path):
        print("=" * 65)
        print("¡Checkpoint detected! Restoring state... / ¡Checkpoint detectado! Restaurando estado...")
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            with open(lora_weights_path, "rb") as f:
                set_peft_model_state_dict(model, load(f.read()))
            optimizer.load_state_dict(torch.load(OPT_FILE, weights_only=False))
            print(f"Resuming training from step / Reanudando entrenamiento desde el paso {start_step}...")
        except Exception as e:
            print(f"[!] Warning reading checkpoint / Advertencia al leer checkpoint: {e}")
            start_step = 0
        print("=" * 65)

    # ── HAND-OFF ENTRE FASES (resolución progresiva) ──────────────────────────
    # Cuando no hay checkpoint propio de esta fase pero se indica init_lora_from,
    # se cargan SÓLO los pesos del adapter de la fase anterior. El optimizador
    # queda fresco (momentos de Adam reiniciados) y start_step=0: cada fase re-warmea
    # sobre la nueva escala de gradientes en vez de arrastrar la inercia anterior.
    if start_step == 0 and INIT_LORA_FROM:
        prev_adapter = os.path.join(INIT_LORA_FROM, "adapter_model.safetensors")
        if os.path.exists(prev_adapter):
            print("=" * 65)
            print(f"Phase hand-off: loading LoRA weights from previous phase / Cargando LoRA de la fase previa:\n  {prev_adapter}")
            with open(prev_adapter, "rb") as f:
                set_peft_model_state_dict(model, load(f.read()))
            print("[OK] LoRA initialized from previous phase; optimizer starts fresh / LoRA inicializada; optimizador desde cero.")
            print("=" * 65)
        else:
            print(f"[!] init_lora_from set but adapter not found / adapter no encontrado: {prev_adapter}")

    last_step_executed = start_step

    def save_checkpoint_now(current_s):
        if current_s <= 0:
            return
        print(f"\nSaving checkpoint state at step / Guardando estado en paso {current_s}...")
        os.makedirs(RESUME_DIR, exist_ok=True)
        model.save_pretrained(RESUME_DIR)
        torch.save(optimizer.state_dict(), OPT_FILE)
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            f.write(str(current_s))
        ckpt = os.path.join(OUTPUT_DIR, f"Krea2_LoRA_step_{current_s}.safetensors")
        _export_lora(model, ckpt)
        print(f"✓ Checkpoint saved successfully at step / Checkpoint guardado en paso {current_s}: {ckpt}")

    def handle_signal(sig, frame):
        nonlocal last_step_executed
        print(f"\n[!] Signal received / Señal de detención recibida ({sig}).")
        save_checkpoint_now(last_step_executed)
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handle_signal)
    except Exception:
        pass

    model.train()
    optimizer.zero_grad(set_to_none=True)

    pin = torch.cuda.is_available()
    cache_data, buckets = {}, defaultdict(list)

    def compact(emb, msk):
        """Se queda sólo con los tokens reales del caption y descarta la máscara.

        Es exacto: los tokens de texto no llevan RoPE (prepare_position_ids les
        asigna la posición 0 a todos) ni en text_fusion, así que la atención es
        equivariante a permutación sobre ellos, y sus salidas se descartan en
        `hidden_states[:, text_seq_len:]`. La máscara sólo servía como
        key-padding, de modo que eliminar los tokens de relleno equivale a
        enmascararlos, y sin máscara SDPA puede usar flash attention.
        """
        idx = msk[0].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:      # caption sin tokens válidos: dejarlo como estaba
            return emb, msk
        return emb[:, idx].contiguous(), None

    for f in os.listdir(CACHE_DIR):
        if f.startswith(".") or not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        lat  = torch.load(f"{CACHE_DIR}/{nombre}_latent.pt", weights_only=True)
        emb  = torch.load(f"{CACHE_DIR}/{nombre}_embed.pt",  weights_only=True)
        msk  = torch.load(f"{CACHE_DIR}/{nombre}_mask.pt",   weights_only=True).bool()
        lat, emb = lat.to(torch.bfloat16), emb.to(torch.bfloat16)
        if COMPACT_TEXT:
            emb, msk = compact(emb, msk)
        if pin:
            lat, emb = lat.pin_memory(), emb.pin_memory()
            if msk is not None:
                msk = msk.pin_memory()
        cache_data[nombre] = (lat, emb, msk)
        buckets[(lat.shape[2], lat.shape[3])].append(nombre)

    neg = None
    if os.path.exists(f"{CACHE_DIR}/_neg_embed.pt"):
        neg_emb = torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True)
        neg_msk = torch.load(f"{CACHE_DIR}/_neg_mask.pt",  weights_only=True).bool()
        if COMPACT_TEXT:
            neg_emb, neg_msk = compact(neg_emb, neg_msk)
        neg = (neg_emb, neg_msk)

    pos_cache = {}
    def get_pos_ids(text_len, lh, lw):
        key = (text_len, lh, lw)
        if key not in pos_cache:
            pos_cache[key] = prepare_position_ids(text_len, lh // 2, lw // 2, "cuda")
        return pos_cache[key]

    all_preview_names = sorted(cache_data.keys())

    def get_preview_sample(step):
        if PREVIEW_CAPTION_MODE == "random":
            return random.choice(all_preview_names)
        elif PREVIEW_CAPTION_MODE == "rotate4":
            idx = (step // max(1, PREVIEW_EVERY)) % min(4, len(all_preview_names))
            return all_preview_names[idx]
        else:
            return all_preview_names[0]

    running_loss, t_step_avg = 0.0, 0.0
    print(f"\nSTARTING TRAINING / ¡ARRANCANDO ENTRENAMIENTO! {len(cache_data)} images in {len(buckets)} buckets.")

    try:
        for step in range(start_step + 1, TOTAL_STEPS + 1):
            last_step_executed = step
            t0 = time.time()

            size = random.choice(list(buckets))
            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
            latents = torch.cat([cache_data[n][0] for n in names]).to("cuda", non_blocking=True)
            embeds  = torch.cat([cache_data[n][1] for n in names]).to("cuda", non_blocking=True)
            masks   = None
            if not COMPACT_TEXT:
                masks = torch.cat([cache_data[n][2] for n in names]).to("cuda", non_blocking=True)

            latent_patched = pack_latents(latents)
            B, seq_img, _ = latent_patched.shape

            sigma  = sample_sigma(B, seq_img, "cuda", shift_cfg)
            noise  = torch.randn_like(latent_patched)
            t_exp  = sigma.view(-1, 1, 1)

            target_dtype = next(model.parameters()).dtype
            noisy = ((1 - t_exp) * latent_patched + t_exp * noise).to(target_dtype)
            target = noise - latent_patched

            pos_ids = get_pos_ids(embeds.shape[1], size[0], size[1])

            pred = model(
                hidden_states=noisy,
                encoder_hidden_states=embeds,
                timestep=sigma,
                position_ids=pos_ids,
                encoder_attention_mask=masks,
                return_dict=False,
            )[0]

            loss = F.mse_loss(pred.float(), target.float()) / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            grad_norm = 0.0
            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()
                for gparam in optimizer.param_groups:
                    gparam["lr"] = lr_at(step)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            t_step     = time.time() - t0
            t_step_avg = t_step if t_step_avg == 0 else 0.1 * t_step + 0.9 * t_step_avg
            eta_s      = (TOTAL_STEPS - step) * t_step_avg
            eta        = f"{int(eta_s//3600):02d}:{int((eta_s%3600)//60):02d}:{int(eta_s%60):02d}"
            pct        = step / TOTAL_STEPS
            barra      = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            
            avg_loss = running_loss / max(1, step - start_step)
            progress_line = (
                f"Step/Paso {step:4d}/{TOTAL_STEPS} [{barra}] {pct*100:5.1f}% | "
                f"Loss {avg_loss:.4f} | gnorm {grad_norm:.3f} | "
                f"lr {lr_at(step):.2e} | {t_step_avg:.2f}s/it | ETA {eta}"
            )
            print(f"\r{progress_line}", end="", flush=True)

            if step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                p_name = get_preview_sample(step)
                lat0, emb0, msk0 = cache_data[p_name]
                print(f"\n  [Preview] Mode: {PREVIEW_CAPTION_MODE} | Sample: {p_name}")
                run_preview(model, scheduler, emb0, msk0, neg,
                            (lat0.shape[2] * 8, lat0.shape[3] * 8), step, shift_cfg)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\nTraining completed! / ¡Entrenamiento finalizado!")
    # Guardar también el resume_checkpoint al terminar: en el pipeline progresivo
    # es el hand-off de pesos que carga la fase siguiente vía init_lora_from.
    save_checkpoint_now(TOTAL_STEPS)
    final = os.path.join(OUTPUT_DIR, "Krea2_FINAL_LoRA.safetensors")
    _export_lora(model, final)
    print(f"✓ Final LoRA saved to / Tu LoRA definitivo está en: {final}")


if __name__ == "__main__":
    train_krea2()