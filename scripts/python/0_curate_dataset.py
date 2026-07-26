# -*- coding: utf-8 -*-
"""
0_curate_dataset.py — Curaduría del dataset por identidad/estilo (previa al entrenamiento)
Dataset curation by identity/style (runs before training)

Puntúa cada imagen del dataset por similitud contra 3 imágenes baseline y
reparte el dataset en dos grupos: buena y baja calificación. Soporta modos:
- face: Reconocimiento de identidad facial (ArcFace)
- clothes: Tipo de ropa / outfit (CLIP visual + caption)
- body-type: Tipo de cuerpo / silueta (Pose/CLIP + caption)
- tattoo: Tatuajes / parches de tinta (Saliencia visual + CLIP + caption)

Lee configuración desde pre_cache_settings.json (clave "curation").
"""
import datetime
import json
import os
import sys

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def from_root(path):
    """Resuelve una ruta relativa contra la raíz del proyecto (absolutas intactas)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

DEFAULTS = {
    "dataset_path": "./dataset",
    "curation": {
        "mode": "face",
        "baselines": [],
        "weight_good": 1.0,
        "weight_bad": 0.5,
    },
}

DIFFERENT_PERSON_FLOOR = 0.25
MIN_SCORES_FOR_THRESHOLD = 4
REPORT_NAME = "curation_report.json"

CONFIG_PATH = os.environ.get("PRECACHE_SETTINGS_PATH",
                             os.path.join(PROJECT_ROOT, "pre_cache_settings.json"))

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} not found, using defaults / No se encontró {CONFIG_PATH}, usando valores por defecto.")

DATASET_PATH = from_root(cfg.get("dataset_path", DEFAULTS["dataset_path"]))

_cur = cfg.get("curation") or {}
MODE = str(_cur.get("mode", "face")).lower()
if MODE not in ("face", "clothes", "body-type", "tattoo"):
    MODE = "face"

BASELINES = list(_cur.get("baselines") or [])
WEIGHT_GOOD = float(_cur.get("weight_good", DEFAULTS["curation"]["weight_good"]))
WEIGHT_BAD = float(_cur.get("weight_bad", DEFAULTS["curation"]["weight_bad"]))
CACHE_NAME = f".curation_cache_{MODE}.npz"


# =============================================================================
# STRATEGY EMBEDDERS / EXTRACTORES DE CARACTERÍSTICAS
# =============================================================================

class FaceEmbedder:
    """Extractor de embeddings ArcFace (InsightFace) para puntuar identidad facial."""

    def __init__(self):
        self._app = None

    def _ensure_loaded(self):
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            print("\n[!] InsightFace no está instalado / is not installed.")
            sys.exit(1)
        print("Cargando modelo facial (CPU)... / Loading face model (CPU)...")
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1)

    def _detect_with_pad_retry(self, img_bgr):
        import cv2
        faces = self._app.get(img_bgr)
        if faces:
            return faces
        for pad in (0.25, 0.5):
            h, w = img_bgr.shape[:2]
            py, px = int(h * pad), int(w * pad)
            padded = cv2.copyMakeBorder(img_bgr, py, py, px, px, cv2.BORDER_REPLICATE)
            faces = self._app.get(padded)
            if faces:
                return faces
        return []

    def embed(self, image_path):
        self._ensure_loaded()
        import cv2
        try:
            with Image.open(image_path) as pil:
                img_bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            print(f"\n[!] No se pudo leer {os.path.basename(image_path)}: {exc}")
            return None
        faces = self._detect_with_pad_retry(img_bgr)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = getattr(largest, "normed_embedding", None)
        return None if emb is None else np.asarray(emb, dtype=np.float32)


class CLIPEmbedder:
    """Extractor visual + texto usando CLIP en CPU."""

    def __init__(self):
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            print("\n[!] transformers no está instalado / is not installed.")
            sys.exit(1)
        print("Cargando modelo CLIP (CPU)... / Loading CLIP model (CPU)...")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model.eval()

    def get_caption_text(self, image_path):
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.isfile(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    def embed_crop_and_caption(self, crop_pil, caption=""):
        self._ensure_loaded()
        import torch
        try:
            inputs = self._processor(images=crop_pil, return_tensors="pt")
            with torch.no_grad():
                img_feats = self._model.get_image_features(**inputs)
                img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
                img_emb = img_feats.cpu().numpy()[0]
        except Exception:
            return None

        txt_emb = None
        if caption:
            try:
                txt_inputs = self._processor(text=[caption[:77]], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    txt_feats = self._model.get_text_features(**txt_inputs)
                    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
                    txt_emb = txt_feats.cpu().numpy()[0]
            except Exception:
                txt_emb = None

        if txt_emb is not None:
            combined = np.concatenate([img_emb * 0.7, txt_emb * 0.3])
            norm = np.linalg.norm(combined)
            return combined / norm if norm > 0 else combined
        return img_emb


class ClothesEmbedder:
    """Extractor de ropa/outfit (CLIP de imagen + caption en CPU)."""

    def __init__(self):
        self.clip = CLIPEmbedder()

    def embed(self, image_path):
        try:
            with Image.open(image_path) as pil:
                rgb_img = pil.convert("RGB")
        except Exception:
            return None
        caption = self.clip.get_caption_text(image_path)
        return self.clip.embed_crop_and_caption(rgb_img, caption)


class BodyEmbedder:
    """Extractor de tipo de cuerpo (silueta/detección + CLIP + caption)."""

    def __init__(self):
        self.clip = CLIPEmbedder()

    def embed(self, image_path):
        import cv2
        try:
            with Image.open(image_path) as pil:
                rgb_img = pil.convert("RGB")
                img_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

        cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        body_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        if len(bodies) > 0:
            x, y, w, h = max(bodies, key=lambda b: b[2] * b[3])
            crop = rgb_img.crop((x, y, x + w, y + h))
        else:
            h, w = img_bgr.shape[:2]
            crop = rgb_img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))

        caption = self.clip.get_caption_text(image_path)
        return self.clip.embed_crop_and_caption(crop, caption)


class TattooEmbedder:
    """Extractor de tatuajes (parches de tinta/saliencia + CLIP + caption)."""

    def __init__(self):
        self.clip = CLIPEmbedder()

    def embed(self, image_path):
        import cv2
        try:
            with Image.open(image_path) as pil:
                rgb_img = pil.convert("RGB")
                img_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        tattoo_region = cv2.bitwise_and(edges, edges, mask=skin_mask)

        contours, _ = cv2.findContours(tattoo_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

        if not valid_contours:
            return None

        all_pts = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        crop = rgb_img.crop((max(0, x - 10), max(0, y - 10), x + w + 10, y + h + 10))

        caption = self.clip.get_caption_text(image_path)
        return self.clip.embed_crop_and_caption(crop, caption)


def get_embedder(mode):
    if mode == "clothes":
        return ClothesEmbedder()
    elif mode == "body-type":
        return BodyEmbedder()
    elif mode == "tattoo":
        return TattooEmbedder()
    else:
        return FaceEmbedder()


# =============================================================================
# CACHÉ DE EMBEDDINGS / EMBEDDING CACHE
# =============================================================================

def fingerprint(path):
    st = os.stat(path)
    txt_path = os.path.splitext(path)[0] + ".txt"
    txt_mtime = int(os.stat(txt_path).st_mtime) if os.path.isfile(txt_path) else 0
    return f"{int(st.st_mtime)}:{st.st_size}:{txt_mtime}"


def load_cache(dataset_dir):
    path = os.path.join(dataset_dir, CACHE_NAME)
    if not os.path.exists(path):
        return {}, {}
    try:
        with np.load(path, allow_pickle=False) as data:
            fps = json.loads(str(data["__fingerprints__"]))
            embs = {k: data[k] for k in data.files if k != "__fingerprints__"}
        return embs, fps
    except Exception:
        return {}, {}


def save_cache(dataset_dir, embs, fps):
    path = os.path.join(dataset_dir, CACHE_NAME)
    tmp = path + ".tmp.npz"
    try:
        np.savez_compressed(tmp, __fingerprints__=np.array(json.dumps(fps)), **embs)
        os.replace(tmp, path)
    except Exception as exc:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# =============================================================================
# CURADURÍA / CURATION
# =============================================================================

def resolve_baseline(name, dataset_dir):
    candidates = [name, os.path.join(dataset_dir, name)]
    for ext in IMAGE_EXTS:
        candidates.append(os.path.join(dataset_dir, name + ext))
    for c in candidates:
        resolved = from_root(c) if not os.path.isabs(c) else c
        if os.path.isfile(resolved):
            return resolved
    return None


def auto_threshold(scores):
    ss = sorted(s for s in scores if s is not None)
    if len(ss) < MIN_SCORES_FOR_THRESHOLD:
        return None
    n = len(ss)
    med = ss[n // 2]
    q1, q3 = ss[n // 4], ss[(3 * n) // 4]
    return max(med - 1.5 * (q3 - q1), DIFFERENT_PERSON_FLOOR)


def curate():
    if not os.path.isdir(DATASET_PATH):
        print(f"[!] La carpeta del dataset no existe / Dataset folder does not exist: {DATASET_PATH}")
        return 1

    files = sorted(f for f in os.listdir(DATASET_PATH)
                   if not f.startswith(".")
                   and os.path.isfile(os.path.join(DATASET_PATH, f))
                   and f.lower().endswith(IMAGE_EXTS))
    if not files:
        print(f"[!] No hay imágenes en / No images found in: {DATASET_PATH}")
        return 1

    if len(BASELINES) != 3:
        print(f"\n[!] Hacen falta exactamente 3 baselines (hay {len(BASELINES)}) / "
              f"exactly 3 baselines required.")
        return 1

    resolved = [(b, resolve_baseline(b, DATASET_PATH)) for b in BASELINES]
    missing = [b for b, p in resolved if p is None]
    if missing:
        print(f"\n[!] No se encontraron estos baselines / baselines not found: {', '.join(missing)}")
        return 1

    print(f"Modo de curaduría activo / Active curation mode: '{MODE.upper()}'")
    embedder = get_embedder(MODE)
    embs, fps = load_cache(DATASET_PATH)
    cache_hits = 0

    def embed_cached(path, key):
        nonlocal cache_hits
        fp = fingerprint(path)
        if key in embs and fps.get(key) == fp:
            cache_hits += 1
            cached = embs[key]
            return None if cached.size == 0 else cached
        emb = embedder.embed(path)
        embs[key] = np.zeros(0, dtype=np.float32) if emb is None else emb
        fps[key] = fp
        return emb

    # ── Baselines ────────────────────────────────────────────────────────────
    base_embs = []
    base_missing = []
    for name, path in resolved:
        key = os.path.basename(path) if os.path.dirname(path) == DATASET_PATH else path
        emb = embed_cached(path, key)
        if emb is None:
            base_missing.append(os.path.basename(path))
        base_embs.append(emb)

    if base_missing:
        print(f"\n[!] No se detectó la característica ({MODE}) en estos baselines: {', '.join(base_missing)}")
        save_cache(DATASET_PATH, embs, fps)
        return 1

    # ── Puntuación / Scoring ─────────────────────────────────────────────────
    print(f"Puntuando {len(files)} imagen(es) contra 3 baselines en modo {MODE.upper()}...")
    scores = {}
    for i, fname in enumerate(files, 1):
        path = os.path.join(DATASET_PATH, fname)
        stem = os.path.splitext(fname)[0]
        emb = embed_cached(path, fname)
        scores[stem] = None if emb is None else float(np.mean([float(np.dot(b, emb))
                                                               for b in base_embs]))
        print(f"\rPuntuando... {i}/{len(files)}", end="", flush=True)
    print()

    save_cache(DATASET_PATH, embs, fps)

    threshold = auto_threshold(scores.values())
    scored = [s for s in scores.values() if s is not None]
    unscored_count = len(scores) - len(scored)

    # ── Informe / Report ─────────────────────────────────────────────────────
    report = {
        "version": 1,
        "mode": MODE,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_path": DATASET_PATH,
        "baselines": [os.path.basename(p) for _, p in resolved],
        "auto_threshold": threshold,
        "weights": {"good": WEIGHT_GOOD, "bad": WEIGHT_BAD},
        "images": {stem: {"score": (None if s is None else round(s, 6))}
                   for stem, s in scores.items()},
    }
    report_path = os.path.join(DATASET_PATH, REPORT_NAME)
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, report_path)

    # ── Resumen / Summary ────────────────────────────────────────────────────
    # Para face: no detectado va a BUENO por defecto.
    # Para clothes, body-type, tattoo: no detectado va a BAJO (low-quality) por defecto.
    if threshold is None:
        if MODE == "face":
            good, bad = len(scores), 0
        else:
            good, bad = len(scored), unscored_count
    else:
        bad = sum(1 for s in scores.values() if (s is None and MODE != "face") or (s is not None and s < threshold))
        good = len(scores) - bad

    print()
    print("=" * 70)
    print(f"  CURADURÍA COMPLETADA ({MODE.upper()}) / CURATION COMPLETE")
    print("=" * 70)
    print(f"  Imágenes puntuadas   : {len(scored)} de {len(files)}"
          + (f" ({cache_hits} desde caché)" if cache_hits else ""))
    if unscored_count:
        fallback_grp = "BUENO" if MODE == "face" else "BAJO (Low Quality)"
        print(f"  Sin área detectada   : {unscored_count} → grupo {fallback_grp}")
    if threshold is None:
        print(f"  Umbral automático    : no calculable -> asignación por defecto según modo")
    else:
        print(f"  Umbral automático    : {threshold * 100:.0f}%")
        print(f"  Buena calificación   : {good} imagen(es) · peso ×{WEIGHT_GOOD}")
        print(f"  Baja calificación    : {bad} imagen(es) · peso ×{WEIGHT_BAD}")
    print(f"  Informe              : {report_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(curate())
