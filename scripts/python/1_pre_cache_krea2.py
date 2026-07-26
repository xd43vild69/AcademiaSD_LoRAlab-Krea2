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
import hashlib
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

# Raíz del proyecto (este script vive en scripts/python/). Todas las rutas se
# anclan aquí en vez de al directorio de trabajo, para que funcione invocado
# desde cualquier sitio.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def from_root(path):
    """Resuelve una ruta relativa contra la raíz del proyecto (absolutas intactas)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


# Carpeta contenedora de todo lo que genera el pre-caché (una sola en la raíz).
CACHE_ROOT = os.path.join(PROJECT_ROOT, "cached_data_local")

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Krea-2-NF4",
    "dataset_path": "./dataset",
    "cache_dir": f"{CACHE_ROOT}/default",
    "target_area": 512 * 512,
    "max_side": 1280,
    "multiple": 16,
    "max_seq_len": 128,
    "project_name": "",
    "trigger_word": "",
    "resolutions": [],

    # ── D1: manifiesto e invalidación de caché ───────────────────────────────
    "cache_key": "mtime",        # "mtime" | "sha1"
    "prune_orphans": "warn",     # "warn" | "delete" | "off"
    "force_recache": False,
    # ── D2: aumento por espejado ─────────────────────────────────────────────
    "flip_x": False,
    # ── C5: prompts de preview ───────────────────────────────────────────────
    "sample_prompts": [],
}

# Se incrementa cuando cambia la normalización o el empaquetado de latentes, para
# invalidar automáticamente todas las cachés existentes.
LATENT_VERSION = 1

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = os.environ.get("PRECACHE_SETTINGS_PATH",
                             os.path.join(PROJECT_ROOT, "pre_cache_settings.json"))

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} not found, using defaults / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID     = cfg.get("model_id",     DEFAULTS["model_id"])
# El modelo local vive en la raíz del proyecto. Se ancla sólo si la carpeta
# existe ahí, para no romper el caso de un repo-id de Hugging Face; sin esto,
# ejecutar desde otro directorio dispararía una descarga completa del modelo.
if not os.path.isabs(MODEL_ID) and os.path.isdir(from_root(MODEL_ID)):
    MODEL_ID = from_root(MODEL_ID)
DATASET_PATH = from_root(cfg.get("dataset_path", DEFAULTS["dataset_path"]))
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

CACHE_KEY     = str(cfg.get("cache_key", DEFAULTS["cache_key"])).strip().lower()
PRUNE_ORPHANS = str(cfg.get("prune_orphans", DEFAULTS["prune_orphans"])).strip().lower()
FORCE_RECACHE = bool(cfg.get("force_recache", DEFAULTS["force_recache"]))
FLIP_X        = bool(cfg.get("flip_x", DEFAULTS["flip_x"]))
SAMPLE_PROMPTS = list(cfg.get("sample_prompts", DEFAULTS["sample_prompts"]) or [])

if CACHE_KEY not in ("mtime", "sha1"):
    print(f"⚠ Invalid cache_key '{CACHE_KEY}'. Using 'mtime' / usando 'mtime'.")
    CACHE_KEY = "mtime"
if PRUNE_ORPHANS not in ("warn", "delete", "off"):
    print(f"⚠ Invalid prune_orphans '{PRUNE_ORPHANS}'. Using 'warn' / usando 'warn'.")
    PRUNE_ORPHANS = "warn"


def res_label(area):
    """Etiqueta corta del subdir a partir del área (ej. 589824 -> '768')."""
    return str(int(round(math.sqrt(area))))

# Formato automático de carpeta de caché según nombre del proyecto.
# Todo lo generado se agrupa bajo CACHE_ROOT para no ensuciar la raíz.
# Sin project_name se respeta un cache_dir explícito (lo usa run_progressive.py
# para apuntar al subdirectorio de resolución de cada fase).
if PROJECT_NAME:
    CACHE_DIR = os.path.join(CACHE_ROOT, PROJECT_NAME)
else:
    CACHE_DIR = from_root(cfg.get("cache_dir", DEFAULTS["cache_dir"]))

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
print(f"  Cache key / Invalidación    : {CACHE_KEY} (prune_orphans={PRUNE_ORPHANS}"
      f"{', force_recache' if FORCE_RECACHE else ''})")
print(f"  Flip augmentation / Espejado: {'ON (duplica el dataset)' if FLIP_X else 'OFF'}")
if SAMPLE_PROMPTS:
    print(f"  Sample prompts / Previews   : {len(SAMPLE_PROMPTS)}")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def free_vram(*tensors):
    """Libera VRAM. Sólo para transiciones de fase, nunca por imagen.

    Ojo con la firma: `del t` borra el nombre local del bucle, no la referencia
    del llamante, así que pasar tensores aquí no los libera —lo único que hace
    es el gc + empty_cache. Se llamaba 12 veces por imagen (~18 ms cada una)
    creyendo que servía para algo, y encima empty_cache destruye el caching
    allocator y obliga a cudaMalloc de nuevo en la iteración siguiente.
    """
    for t in tensors:
        del t
    gc.collect()
    torch.cuda.empty_cache()


def save_tensor(tensor, path):
    """torch.save borrando antes el destino.

    El borrado importa desde que se usan hardlinks: sin él, `torch.save` abriría
    en modo 'wb' el inodo compartido y una escritura a medias corrompería todas
    las resoluciones a la vez, en vez de sólo ésta.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    torch.save(tensor, path)


def link_or_copy(source, dest, tensor):
    """Enlaza `dest` a `source` (mismo contenido) o, si no se puede, lo escribe.

    El fallback cubre Windows sin privilegios, FAT y cachés repartidas entre
    discos distintos; en esos casos se vuelve al comportamiento de siempre.
    """
    try:
        os.remove(dest)
    except FileNotFoundError:
        pass
    try:
        os.link(source, dest)
    except OSError:
        torch.save(tensor, dest)


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


# ── D1: MANIFIESTO DE CACHÉ / CACHE MANIFEST ────────────────────────────────

def manifest_config():
    """Ajustes que, si cambian, invalidan todos los latentes ya cacheados."""
    return {
        "multiple": MULTIPLE,
        "max_side": MAX_SIDE,
        "max_seq_len": MAX_SEQ_LEN,
        "trigger_word": TRIGGER_WORD,
        "flip_x": FLIP_X,
        "latent_version": LATENT_VERSION,
    }


def load_manifest(out_dir):
    path = os.path.join(out_dir, "cache_manifest.json")
    if not os.path.exists(path):
        return {"version": 1, "config": {}, "entries": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("config") != manifest_config():
            print(f"   [i] Cache config changed for {os.path.basename(out_dir)}; "
                  f"re-encoding all / la configuración cambió; se reencoda todo.")
            return {"version": 1, "config": {}, "entries": {}}
        return data
    except Exception:
        return {"version": 1, "config": {}, "entries": {}}


def save_manifest(out_dir, entries):
    data = {"version": 1, "config": manifest_config(), "entries": entries}
    path = os.path.join(out_dir, "cache_manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def file_fingerprint(path):
    """Huella de la imagen fuente: mtime+size (rápido) o sha1 (robusto ante copias)."""
    stat = os.stat(path)
    if CACHE_KEY == "sha1":
        digest = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return {"sha1": digest.hexdigest(), "src_size": stat.st_size}
    return {"src_mtime": int(stat.st_mtime), "src_size": stat.st_size}


def caption_fingerprint(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def prune_orphans(out_dir, valid_names):
    """Latentes de imágenes borradas o renombradas.

    Sin esto se quedaban en la caché para siempre y el entrenador los seguía
    entrenando en silencio; con el sampler por épocas además tendrían un turno
    garantizado cada época.
    """
    present = {f[:-len("_latent.pt")] for f in os.listdir(out_dir)
               if f.endswith("_latent.pt") and not f.startswith(".")}
    orphans = sorted(present - set(valid_names))
    if not orphans:
        return
    label = os.path.basename(out_dir)
    if PRUNE_ORPHANS == "delete":
        for name in orphans:
            for suffix in ("_latent.pt", "_embed.pt", "_mask.pt"):
                try:
                    os.remove(os.path.join(out_dir, f"{name}{suffix}"))
                except OSError:
                    pass
        print(f"   [OK] Pruned {len(orphans)} orphan entries from {label} / eliminados.")
    elif PRUNE_ORPHANS == "warn":
        print(f"\n[!] {len(orphans)} orphan cache entries in {label} "
              f"(source image deleted or renamed) / huérfanos:")
        for name in orphans[:10]:
            print(f"      {name}")
        if len(orphans) > 10:
            print(f"      ... and {len(orphans) - 10} more")
        print("[!] They will still be trained on. Set prune_orphans='delete' to remove them "
              "/ se seguirán entrenando.")


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
        # Sin free_vram aquí: no liberaba nada (img_tensor y z siguen vivos en
        # este frame hasta el return) y costaba ~18 ms por llamada. Al salir de
        # la función el refcount los suelta y el caching allocator reutiliza los
        # bloques, que es justo lo que empty_cache impedía.
        return bw, bh, latent

    manifests = {out_dir: ({} if FORCE_RECACHE else load_manifest(out_dir))
                 for out_dir, _ in targets}
    reused = encoded = 0

    with torch.inference_mode():
        # El embed negativo es independiente de la resolución: se guarda en cada caché.
        neg_embed, neg_mask = pipe.encode_prompt(prompt="", max_sequence_length=MAX_SEQ_LEN)
        for out_dir, _ in targets:
            torch.save(neg_embed.cpu(), os.path.join(out_dir, "_neg_embed.pt"))
            torch.save(neg_mask.cpu(),  os.path.join(out_dir, "_neg_mask.pt"))
        del neg_embed, neg_mask

        # ── C5: prompts de preview ──────────────────────────────────────────
        # El entrenador carga la pipeline con text_encoder=None y no puede
        # codificar texto arbitrario, así que se pre-codifican aquí igual que el
        # embed negativo. Coste cero de VRAM en el entrenamiento.
        for out_dir, _ in targets:
            for old in os.listdir(out_dir):
                if old.startswith("_sample") and old.endswith((".pt", ".json")):
                    os.remove(os.path.join(out_dir, old))
        for i, item in enumerate(SAMPLE_PROMPTS):
            spec = {"prompt": item} if isinstance(item, str) else dict(item)
            text = str(spec.get("prompt", "")).strip()
            if not text:
                continue
            if TRIGGER_WORD and TRIGGER_WORD.lower() not in text.lower():
                text = f"{TRIGGER_WORD}, {text}".strip(", ")
            s_emb, s_msk = pipe.encode_prompt(prompt=text, max_sequence_length=MAX_SEQ_LEN)
            spec["prompt"] = text
            for out_dir, _ in targets:
                torch.save(s_emb.cpu(), os.path.join(out_dir, f"_sample{i}_embed.pt"))
                torch.save(s_msk.cpu(), os.path.join(out_dir, f"_sample{i}_mask.pt"))
                with open(os.path.join(out_dir, f"_sample{i}_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(spec, f, indent=2, ensure_ascii=False)
            del s_emb, s_msk
            print(f"   [OK] Sample prompt {i} cached / cacheado: {text[:60]}")

        for idx, archivo in enumerate(archivos_img, 1):
            nombre_base = os.path.splitext(archivo)[0]
            ruta_texto  = os.path.join(DATASET_PATH, f"{nombre_base}.txt")
            ruta_img    = os.path.join(DATASET_PATH, archivo)

            prompt = ""
            if os.path.exists(ruta_texto):
                with open(ruta_texto, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
                prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")

            # ── D1: ¿se puede reutilizar lo ya cacheado? ────────────────────
            entry = dict(file_fingerprint(ruta_img))
            entry["caption_sha1"] = caption_fingerprint(prompt)

            def is_cached(out_dir):
                previous = manifests[out_dir].get("entries", {}).get(nombre_base)
                if previous is None:
                    return False
                if {k: previous.get(k) for k in entry} != entry:
                    return False
                names = [nombre_base] + ([f"{nombre_base}__flip"] if FLIP_X else [])
                return all(os.path.exists(os.path.join(out_dir, f"{n}{suffix}"))
                           for n in names for suffix in ("_latent.pt", "_embed.pt", "_mask.pt"))

            pending = [(out_dir, area) for out_dir, area in targets if not is_cached(out_dir)]
            if not pending:
                reused += 1
                print(f"[{idx}/{len(archivos_img)}] Up to date / Sin cambios: {archivo}")
                for out_dir, _ in targets:
                    manifests[out_dir].setdefault("entries", {})[nombre_base] = entry
                continue

            print(f"[{idx}/{len(archivos_img)}] Processing / Procesando: {archivo}")
            try:
                img = Image.open(ruta_img).convert("RGB")
            except Exception as err:
                print(f"[!] Warning: Cannot open image / No se pudo abrir la imagen '{archivo}': {err}")
                continue

            # ── 1. TEXTO → EMBEDDINGS (independiente de la resolución) ──────
            embeds, mask = pipe.encode_prompt(prompt=prompt, max_sequence_length=MAX_SEQ_LEN)
            embeds_cpu, mask_cpu = embeds.cpu(), mask.cpu()
            del embeds, mask

            # ── 2. IMAGEN → LATENTES (VAE), una vez por resolución ──────────
            # D2: el espejado se cachea como una entrada más (`nombre__flip`). El
            # entrenador la recoge en su escaneo de directorio sin cambio alguno,
            # y el sampler por épocas garantiza que se vean original y espejo.
            variants = [("", img)]
            if FLIP_X:
                variants.append(("__flip", img.transpose(Image.FLIP_LEFT_RIGHT)))

            sizes = []
            # embeds_cpu y mask_cpu se calculan una vez por imagen y no dependen
            # de la resolución: el mismo fichero se escribía idéntico en cada
            # subdir (~7.6 MB por imagen y resolución). Se escribe una vez y las
            # demás resoluciones se enlazan al mismo inodo. El trainer no se
            # entera: torch.load abre un hardlink igual que cualquier fichero.
            #
            # La clave incluye el sufijo a propósito, aunque el tensor de la
            # variante __flip sea el mismo objeto: torch.save incrusta el nombre
            # del fichero como prefijo dentro del zip ('an_12_embed/data.pkl'),
            # así que enlazar entre nombres distintos daría el mismo tensor pero
            # NO los mismos bytes que escribía el código anterior. Enlazando sólo
            # entre resoluciones —donde el nombre base coincide— la caché queda
            # byte a byte igual que antes.
            first_written = {}
            for out_dir, area in pending:
                for suffix, variant in variants:
                    bw, bh, latent = vae_latent(variant, area)
                    name = f"{nombre_base}{suffix}"
                    save_tensor(latent, os.path.join(out_dir, f"{name}_latent.pt"))
                    for kind, tensor in (("embed", embeds_cpu), ("mask", mask_cpu)):
                        dest = os.path.join(out_dir, f"{name}_{kind}.pt")
                        source = first_written.get((suffix, kind))
                        if source is None:
                            save_tensor(tensor, dest)
                            first_written[(suffix, kind)] = dest
                        else:
                            link_or_copy(source, dest, tensor)
                sizes.append(f"{bw}x{bh}")

            for out_dir, _ in targets:
                manifests[out_dir].setdefault("entries", {})[nombre_base] = entry
            encoded += 1
            flip_note = " (+flip)" if FLIP_X else ""
            print(f"   [OK] {' · '.join(sizes)}{flip_note} | Latents & Embeddings cached "
                  f"/ Latentes y embeddings cacheados.")

    # El manifiesto se reconstruye a partir de las imágenes realmente presentes:
    # arrastrar las entradas antiguas haría que una imagen borrada o renombrada
    # siguiera contando como válida y su latente nunca se reportase como huérfano.
    present_names = {os.path.splitext(f)[0] for f in archivos_img}
    for out_dir, _ in targets:
        entries = {name: data
                   for name, data in manifests[out_dir].get("entries", {}).items()
                   if name in present_names}
        save_manifest(out_dir, entries)
        valid = list(entries) + ([f"{n}__flip" for n in entries] if FLIP_X else [])
        prune_orphans(out_dir, valid)

    free_vram(pipe)
    print(f"\n✓ Pre-caching finished! {encoded} encoded, {reused} reused, VRAM freed "
          f"/ ¡Pre-caché finalizado! {encoded} codificadas, {reused} reutilizadas.")


if __name__ == "__main__":
    preprocess_krea2()