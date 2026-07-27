# -*- coding: utf-8 -*-
"""
recaption_dataset.py — Auto-recaption con Qwen3-VL bajo demanda desde el tab de Curation

Regenera captions con IA sobre: una imagen (target="single"), las imágenes por
debajo del umbral de curación (target="below_threshold"), o todo el dataset
(target="all"). Lee configuración desde recaption_settings.json (sidecar,
escrito por POST /api/save-recaption justo antes de lanzar este script vía
POST /api/run — mismo patrón que pre_cache_settings.json/train_settings.json).

No toca el training loop: es 100% manual, disparado desde un botón en Curation,
a diferencia del auto-recaption-entre-épocas de Fizgig (atado a loss_watch).
"""
import json
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Raíz del proyecto (este script vive en scripts/python/). Todas las rutas se
# anclan aquí en vez de al directorio de trabajo.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def from_root(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "recaption_settings.json")
CURATION_REPORT_NAME = "curation_report.json"
CURATION_OVERRIDES_NAME = "curation_overrides.json"

DEFAULTS = {
    "dataset_path": "./dataset",
    "model_id": "Krea-2-NF4",
    "target": "all",           # "single" | "below_threshold" | "all"
    "filename": "",
    "threshold": None,
    "detailed": False,
    "include_trigger": True,
    "trigger_word": "",
    "keep_backup": True,
}


def load_settings():
    if not os.path.isfile(SETTINGS_FILE):
        print(f"[!] No se encontró {SETTINGS_FILE}")
        return dict(DEFAULTS)
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(cfg)
    return merged


# ── Resolución de grupo good/bad ────────────────────────────────────────────
# Misma lógica pequeña que server.py::resolve_curation_group y
# 2_train_lora_krea2.py::_curation_group. CLAUDE.md ya documenta que se
# replica deliberadamente en 3 sitios para que un reescaneo nunca pise
# decisiones manuales; este script es la 4ª réplica.
def resolve_curation_group(score, threshold, override, mode="face"):
    if override in ("good", "bad"):
        return override
    default_group = "good" if mode == "face" else "bad"
    if score is None or threshold is None:
        return default_group
    return "good" if score >= threshold else "bad"


def load_curation_scores(dataset_dir):
    """Devuelve {stem: score_efectivo_o_None} usando curation_report.json +
    curation_overrides.json, o {} si nunca se curó el dataset."""
    report_path = os.path.join(dataset_dir, CURATION_REPORT_NAME)
    overrides_path = os.path.join(dataset_dir, CURATION_OVERRIDES_NAME)
    if not os.path.isfile(report_path):
        return {}, None

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    overrides = {}
    if os.path.isfile(overrides_path):
        with open(overrides_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)

    images = report.get("images") or {}
    auto_threshold = report.get("auto_threshold")
    manual = overrides.get("threshold")
    threshold = manual if isinstance(manual, (int, float)) else auto_threshold
    group_overrides = overrides.get("groups") or {}

    scores = {}
    for stem, entry in images.items():
        scores[stem] = entry.get("score")
    return scores, threshold, group_overrides


def _atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def resolve_targets(dataset_dir, cfg):
    """Lista de rutas base (sin extensión) de las imágenes a recaptionar."""
    all_files = sorted(
        f for f in os.listdir(dataset_dir)
        if not f.startswith(".") and os.path.isfile(os.path.join(dataset_dir, f))
        and f.lower().endswith(IMAGE_EXTS)
    )

    target = cfg.get("target", "all")

    if target == "single":
        filename = cfg.get("filename", "")
        if filename and filename in all_files:
            return [filename]
        print(f"[!] Imagen no encontrada en el dataset: {filename}")
        return []

    if target == "below_threshold":
        scores, auto_threshold, group_overrides = load_curation_scores(dataset_dir)
        threshold = cfg.get("threshold")
        if threshold is None:
            threshold = auto_threshold
        if threshold is None:
            print("[!] No hay umbral de curación disponible; corre Curation primero.")
            return []

        selected = []
        for f in all_files:
            stem = os.path.splitext(f)[0]
            score = scores.get(stem)
            override = group_overrides.get(stem)
            group = resolve_curation_group(score, threshold, override)
            if group == "bad":
                selected.append(f)
        return selected

    # target == "all"
    return all_files


def run():
    cfg = load_settings()
    dataset_dir = from_root(cfg.get("dataset_path", "./dataset"))

    if not os.path.isdir(dataset_dir):
        print(f"[!] La carpeta del dataset no existe: {dataset_dir}")
        return 1

    targets = resolve_targets(dataset_dir, cfg)
    if not targets:
        print("[!] No hay imágenes que recaptionar con los criterios dados.")
        return 1

    print(f"[*] Recaption Qwen3-VL: {len(targets)} imagen(es) objetivo ({cfg.get('target')})")
    print(f"[*] Estilo: {'detallado' if cfg.get('detailed') else 'conciso'} | "
          f"Trigger word: {'sí' if cfg.get('include_trigger') and cfg.get('trigger_word') else 'no'} | "
          f"Backup: {'sí' if cfg.get('keep_backup') else 'no'}")

    print("[*] Cargando Qwen3-VL-4B (text encoder local, modo generación)...")
    import torch
    from caption_qwen3vl import generate_caption, load_captioner

    model_dir = from_root(cfg.get("model_id", "Krea-2-NF4"))
    text_encoder_dir = os.path.join(model_dir, "text_encoder")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, processor = load_captioner(text_encoder_dir, device=device, dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        # La GPU puede estar compartida con otro proceso (ComfyUI, otra sesión de
        # entrenamiento, etc.) — en vez de morir, se reintenta en CPU. Más lento
        # (minutos por imagen), pero siempre termina en vez de fallar el run entero.
        if device == "cpu":
            raise
        print("[!] VRAM insuficiente en GPU (probablemente compartida con otro proceso). "
              "Reintentando en CPU — más lento, puede tardar varios minutos por imagen...")
        torch.cuda.empty_cache()
        device = "cpu"
        model, processor = load_captioner(text_encoder_dir, device=device, dtype=torch.bfloat16)
    print(f"[OK] Modelo cargado ({device}).")

    trigger_word = str(cfg.get("trigger_word", "")).strip()
    include_trigger = bool(cfg.get("include_trigger")) and bool(trigger_word)
    keep_backup = bool(cfg.get("keep_backup"))
    detailed = bool(cfg.get("detailed"))

    total = len(targets)
    ok_count = 0
    for i, filename in enumerate(targets, 1):
        img_path = os.path.join(dataset_dir, filename)
        stem = os.path.splitext(filename)[0]
        txt_path = os.path.join(dataset_dir, stem + ".txt")
        bak_path = txt_path + ".bak"

        try:
            caption = generate_caption(model, processor, img_path, detailed=detailed)
            if include_trigger and trigger_word.lower() not in caption.lower():
                caption = f"{trigger_word}, {caption}".strip(", ")

            if keep_backup and os.path.isfile(txt_path) and not os.path.isfile(bak_path):
                shutil.copyfile(txt_path, bak_path)

            _atomic_write_text(txt_path, caption)
            ok_count += 1
            print(f"[*] Recaption {i}/{total}: {filename}")
        except Exception as exc:
            print(f"[!] Error recaptionando {filename}: {exc}")

    print(f"[OK] Recaption completado: {ok_count}/{total} imágenes actualizadas.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(run())
