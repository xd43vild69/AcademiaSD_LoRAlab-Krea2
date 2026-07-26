# -*- coding: utf-8 -*-
"""
0_curate_dataset.py — Curaduría del dataset por identidad facial (previa al entrenamiento)
Dataset curation by face identity (runs before training)

Puntúa cada imagen del dataset por similitud ArcFace contra 3 imágenes baseline y
reparte el dataset en dos grupos: buena y baja calificación. El entrenador aplica
un peso distinto a cada grupo (ver curation_weights en 2_train_lora_krea2.py).
Nada se borra ni se mueve: el grupo de baja calificación sigue entrenando, atenuado.

Scores every dataset image by ArcFace similarity against 3 baseline images and splits
the dataset into two groups: good and low rating. The trainer applies a different
weight to each group. Nothing is deleted or moved.

El prefijo 0_ indica que corre ANTES del pre-caché: decide con qué peso entrenará
cada imagen, no qué se cachea. Es CPU-only, así que no compite por VRAM.

Lee configuración desde pre_cache_settings.json (clave "curation").
Reads configuration from pre_cache_settings.json (key "curation").
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

# Raíz del proyecto (este script vive en scripts/python/). Todas las rutas se
# anclan aquí en vez de al directorio de trabajo, para que funcione invocado
# desde cualquier sitio.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def from_root(path):
    """Resuelve una ruta relativa contra la raíz del proyecto (absolutas intactas)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "dataset_path": "./dataset",
    "curation": {
        "baselines": [],       # 3 imágenes que mejor representan el look buscado
        "weight_good": 1.0,
        "weight_bad": 0.5,
    },
}

# El piso del corte automático: por debajo de ~0.25 de similitud coseno, ArcFace
# ya no considera que sean la misma persona. Sirve de suelo para que un dataset
# muy disperso no acabe mandando imágenes válidas al grupo bajo.
DIFFERENT_PERSON_FLOOR = 0.25

# Mínimo de caras puntuadas para que la estadística de outliers signifique algo.
MIN_SCORES_FOR_THRESHOLD = 4

REPORT_NAME = "curation_report.json"
CACHE_NAME = ".curation_cache.npz"

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

DATASET_PATH = from_root(cfg.get("dataset_path", DEFAULTS["dataset_path"]))

_cur = cfg.get("curation") or {}
BASELINES = list(_cur.get("baselines") or [])
WEIGHT_GOOD = float(_cur.get("weight_good", DEFAULTS["curation"]["weight_good"]))
WEIGHT_BAD = float(_cur.get("weight_bad", DEFAULTS["curation"]["weight_bad"]))


# =============================================================================
# EMBEDDINGS FACIALES / FACE EMBEDDINGS
# =============================================================================

class FaceEmbedder:
    """Extractor de embeddings ArcFace (InsightFace) para puntuar identidad.

    Carga los módulos de detección + reconocimiento en CPU: la curaduría corre
    antes del entrenamiento pero no debe competir por VRAM con nada. Los
    embeddings salen ya normalizados en L2 (`normed_embedding`), así que la
    similitud coseno entre dos caras es un producto punto directo.
    """

    def __init__(self):
        self._app = None

    def _ensure_loaded(self):
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            print("\n[!] InsightFace no está instalado / is not installed.")
            print("    Ejecuta ./install_LoRAlab-Krea2.sh, o instala a mano:")
            print("    pip install insightface onnxruntime opencv-python\n")
            sys.exit(1)
        print("Cargando modelo facial (la primera vez descarga ~300 MB)... / "
              "Loading face model (first run downloads ~300 MB)...")
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        # ctx_id=-1 fuerza CPU.
        self._app.prepare(ctx_id=-1)

    def _detect_with_pad_retry(self, img_bgr):
        """Detecta caras, reintentando con un borde añadido si no encuentra ninguna.

        RetinaFace (buffalo_l) falla con caras que LLENAN el encuadre: un primer
        plano es demasiado grande para sus escalas de anclaje y devuelve vacío.
        Añadir un borde reduce la fracción del encuadre que ocupa la cara y la
        devuelve al rango útil del detector; desplaza coordenadas pero no el
        contenido, así que el embedding no cambia. Sólo se añade cuando la pasada
        sin borde falla, así que el encuadre normal no sufre regresión.

        Sin esto se pierden justo los primeros planos, que suelen ser las mejores
        imágenes del dataset.
        """
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
        """Embedding de la cara MÁS GRANDE de la imagen, o None si no hay ninguna.

        Carga con PIL (soporta rutas unicode, a diferencia de cv2.imread) y
        convierte al array BGR que espera InsightFace.
        """
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


# =============================================================================
# CACHÉ DE EMBEDDINGS / EMBEDDING CACHE
# =============================================================================

def fingerprint(path):
    """Huella de la imagen: mtime+tamaño. Mismo criterio que file_fingerprint()
    del pre-caché — barato y suficiente para saber si hay que re-embeber."""
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def load_cache(dataset_dir):
    """Embeddings ya calculados, indexados por nombre de archivo.

    Devuelve (embeddings, fingerprints). Un embedding de longitud 0 significa
    "ya se miró y no hay cara" — se cachea igual que un acierto, para no repetir
    la detección (que es la parte lenta) en cada re-scan.
    """
    path = os.path.join(dataset_dir, CACHE_NAME)
    if not os.path.exists(path):
        return {}, {}
    try:
        with np.load(path, allow_pickle=False) as data:
            fps = json.loads(str(data["__fingerprints__"]))
            embs = {k: data[k] for k in data.files if k != "__fingerprints__"}
        return embs, fps
    except Exception:
        print("[i] Caché de embeddings ilegible; se recalcula / unreadable cache, recomputing.")
        return {}, {}


def save_cache(dataset_dir, embs, fps):
    path = os.path.join(dataset_dir, CACHE_NAME)
    tmp = path + ".tmp.npz"
    try:
        np.savez_compressed(tmp, __fingerprints__=np.array(json.dumps(fps)), **embs)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"[i] No se pudo guardar la caché de embeddings ({exc}) — no es crítico.")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# =============================================================================
# CURADURÍA / CURATION
# =============================================================================

def resolve_baseline(name, dataset_dir):
    """Un baseline puede venir como stem ('img_003'), nombre completo
    ('img_003.png') o ruta. Se admite fuera del dataset: una foto de referencia
    limpia es un baseline perfectamente válido aunque no se entrene."""
    candidates = [name, os.path.join(dataset_dir, name)]
    for ext in IMAGE_EXTS:
        candidates.append(os.path.join(dataset_dir, name + ext))
    for c in candidates:
        resolved = from_root(c) if not os.path.isabs(c) else c
        if os.path.isfile(resolved):
            return resolved
    return None


def auto_threshold(scores):
    """Corte automático: la valla de outliers bajos del propio dataset.

        cutoff = max(mediana − 1.5·IQR, 0.25)

    Un umbral absoluto no sirve porque cada dataset tiene su propia dispersión:
    uno muy homogéneo dejaría el grupo bajo vacío y uno disperso mandaría medio
    dataset abajo. La valla robusta se adapta; el piso evita que un dataset malo
    normalice su propia deriva.

    Devuelve None cuando no hay caras suficientes para que la estadística
    signifique algo — en ese caso todo va al grupo bueno.
    """
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

    # Sólo el nivel superior, igual que el pre-caché: lo que hay en subcarpetas
    # no se entrena, así que tampoco se puntúa.
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
        print("    Elige en la UI las 3 imágenes que mejor representan el look que buscas,")
        print("    o añádelas a mano en pre_cache_settings.json:")
        print('      "curation": { "baselines": ["img_003", "img_017", "img_042"] }')
        print("\n    Se usan 3 y se promedia porque una sola foto hornea su propio sesgo de")
        print("    ángulo, expresión e iluminación en la puntuación de todas las demás.\n")
        return 1

    resolved = [(b, resolve_baseline(b, DATASET_PATH)) for b in BASELINES]
    missing = [b for b, p in resolved if p is None]
    if missing:
        print(f"\n[!] No se encontraron estos baselines / baselines not found: {', '.join(missing)}")
        return 1

    embedder = FaceEmbedder()
    embs, fps = load_cache(DATASET_PATH)
    cache_hits = 0

    def embed_cached(path, key):
        """Embedding vía caché por (mtime, tamaño). La carga del modelo y la
        detección son la parte lenta, así que cambiar de baselines o mover el
        umbral vuelve a puntuar en segundos."""
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
    base_missing_face = []
    for name, path in resolved:
        # Se cachean bajo su ruta absoluta si viven fuera del dataset, para no
        # colisionar con una imagen del dataset que se llame igual.
        key = os.path.basename(path) if os.path.dirname(path) == DATASET_PATH else path
        emb = embed_cached(path, key)
        if emb is None:
            base_missing_face.append(os.path.basename(path))
        base_embs.append(emb)

    if base_missing_face:
        print(f"\n[!] No se detectó cara en estos baselines / no face in baselines: "
              f"{', '.join(base_missing_face)}")
        print("    Elige imágenes con una cara clara y bien visible.\n")
        save_cache(DATASET_PATH, embs, fps)
        return 1

    # ── Puntuación / Scoring ─────────────────────────────────────────────────
    print(f"Puntuando {len(files)} imagen(es) contra 3 baselines... / Scoring...")
    scores = {}
    for i, fname in enumerate(files, 1):
        path = os.path.join(DATASET_PATH, fname)
        stem = os.path.splitext(fname)[0]
        emb = embed_cached(path, fname)
        # El promedio de las 3 similitudes equivale a la similitud contra el
        # centroide (sin normalizar) de los baselines: ninguna foto suelta puede
        # dominar la puntuación con su encuadre particular.
        scores[stem] = None if emb is None else float(np.mean([float(np.dot(b, emb))
                                                               for b in base_embs]))
        print(f"\rPuntuando... {i}/{len(files)}", end="", flush=True)
    print()

    save_cache(DATASET_PATH, embs, fps)

    threshold = auto_threshold(scores.values())
    scored = [s for s in scores.values() if s is not None]
    no_face = len(scores) - len(scored)

    # ── Informe / Report ─────────────────────────────────────────────────────
    # Vive en la carpeta del dataset a propósito: esto es conocimiento del
    # dataset, viaja con las imágenes y sobrevive a renombrar el proyecto.
    # Sólo lleva puntuaciones; el umbral efectivo y las reasignaciones manuales
    # viven en curation_overrides.json, que es propiedad de la UI y NO se
    # sobrescribe al volver a puntuar.
    report = {
        "version": 1,
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
    # Las imágenes sin cara van al grupo BUENO: no puntuable no es lo mismo que
    # mala (un plano de espalda o un perfil extremo son material legítimo).
    if threshold is None:
        good = len(scores)
        bad = 0
    else:
        bad = sum(1 for s in scores.values() if s is not None and s < threshold)
        good = len(scores) - bad

    print()
    print("=" * 70)
    print("  CURADURÍA COMPLETADA / CURATION COMPLETE")
    print("=" * 70)
    print(f"  Imágenes puntuadas   : {len(scored)} de {len(scores)}"
          + (f" ({cache_hits} desde caché)" if cache_hits else ""))
    if no_face:
        print(f"  Sin cara detectada   : {no_face} → grupo BUENO (no puntuable ≠ mala)")
    if threshold is None:
        print(f"  Umbral automático    : no calculable (hacen falta {MIN_SCORES_FOR_THRESHOLD} "
              f"caras) → todo al grupo bueno")
    else:
        print(f"  Umbral automático    : {threshold * 100:.0f}%")
        print(f"  Buena calificación   : {good} imagen(es) · peso ×{WEIGHT_GOOD}")
        print(f"  Baja calificación    : {bad} imagen(es) · peso ×{WEIGHT_BAD}")
    print(f"  Informe              : {report_path}")
    print("=" * 70)
    print("\nAjusta el umbral y reasigna imágenes en la UI antes de entrenar. /")
    print("Adjust the threshold and reassign images in the UI before training.\n")
    return 0


if __name__ == "__main__":
    sys.exit(curate())
