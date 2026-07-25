# -*- coding: utf-8 -*-
"""
run_progressive.py — Orquestador de entrenamiento de resolución progresiva.
Progressive-resolution training orchestrator for Krea 2.

Ejecuta el entrenamiento LoRA por fases (p. ej. 512² → 768² → 1024²) como
subprocesos de PyTorch independientes y secuenciales. Cada fase arranca con un
proceso limpio (asignador CUDA fresco, sin fragmentación acumulada), inicializa
la LoRA con los pesos de la fase anterior (init_lora_from) y reinicia el
optimizador. La resolución de cada fase sale de los subdirectorios que generó
1_pre_cache_krea2.py.

Lee train_settings.json y no lo modifica: escribe un fichero de settings por
fase y lo pasa al entrenador vía la env-var TRAIN_SETTINGS_PATH.
"""
import os
import sys
import json
import time
import signal
import shutil
import subprocess
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# BASE_DIR es scripts/python/; PROJECT_ROOT es la raíz del proyecto, contra la
# que se resuelven configuración y datos generados.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

TRAIN_SCRIPT = BASE_DIR / "2_train_lora_krea2.py"
CONFIG_PATH = os.environ.get("TRAIN_SETTINGS_PATH", str(PROJECT_ROOT / "train_settings.json"))
PHASE_SETTINGS_DIR = PROJECT_ROOT / ".progressive_phases"

# Carpetas contenedoras: todo lo generado se agrupa aquí en vez de en la raíz.
CACHE_ROOT = str(PROJECT_ROOT / "cached_data_local")
OUTPUT_ROOT = str(PROJECT_ROOT / "output_local")

# ── PRESETS DE FASES ─────────────────────────────────────────────────────────
# label: subdir de la caché (lo genera el pre-cache con round(sqrt(area))).
# portion: fracción de total_steps dedicada a la fase.
# gc: gradient_checkpointing (True = seguro/lento, False = rápido/más VRAM).
# Los flags de checkpointing por fase son conservadores por defecto (True);
# se pueden afinar tras medir la VRAM real de cada resolución.
PRESETS = {
    "768_1024": [
        {"label": "768",  "portion": 0.70, "gc": True},
        {"label": "1024", "portion": 0.30, "gc": True},
    ],
    "512_768_1024": [
        {"label": "512",  "portion": 0.40, "gc": True},
        {"label": "768",  "portion": 0.30, "gc": True},
        {"label": "1024", "portion": 0.30, "gc": True},
    ],
}

_current_child = None
_stop_requested = False


def _forward_signal(signum, frame):
    """Reenvía la señal de parada al entrenador hijo para que guarde checkpoint."""
    global _stop_requested
    _stop_requested = True
    child = _current_child
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signal.SIGINT)
        except Exception:
            try:
                child.terminate()
            except Exception:
                pass


def derive_dirs(cfg):
    """Reproduce la lógica project_name → carpetas de los scripts."""
    def anchor(path):
        """Las rutas relativas del JSON se resuelven contra la raíz del proyecto."""
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(str(PROJECT_ROOT), path))

    project = str(cfg.get("project_name", "")).strip()
    if project:
        cache_base = os.path.join(CACHE_ROOT, project)
        output_base = os.path.join(OUTPUT_ROOT, project)
    else:
        cache_base = anchor(cfg.get("cache_dir", os.path.join(CACHE_ROOT, "default")))
        output_base = anchor(cfg.get("output_dir", os.path.join(OUTPUT_ROOT, "default")))
    return cache_base, output_base


def main():
    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)

    if not os.path.exists(CONFIG_PATH):
        print(f"[!] {CONFIG_PATH} not found / no encontrado.")
        return 1

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Sidecar avanzado: se congela en los JSON de fase para que cada
    # .progressive_phases/*.json quede autodescriptivo y reproducible aunque
    # train_advanced.json cambie a mitad del run. train_settings.json manda.
    advanced_path = os.environ.get("TRAIN_ADVANCED_PATH",
                                   os.path.join(str(PROJECT_ROOT), "train_advanced.json"))
    if os.path.exists(advanced_path):
        try:
            with open(advanced_path, "r", encoding="utf-8") as f:
                adv = json.load(f)
            cfg = {**adv, **cfg}
            print(f"[OK] Advanced settings merged from {advanced_path} ({len(adv)} keys)")
        except Exception as exc:
            print(f"[!] Could not read {advanced_path}: {exc} — ignoring / ignorando.")

    preset = str(cfg.get("progressive", "off")).strip().lower()
    if preset in ("off", "", "none"):
        print("[!] 'progressive' is off; nothing to orchestrate. Run the trainer directly.")
        return 1
    if preset not in PRESETS:
        print(f"[!] Unknown progressive preset '{preset}'. Options: {list(PRESETS)}")
        return 1

    phases = [dict(p) for p in PRESETS[preset]]
    portions = cfg.get("phase_portions")
    if isinstance(portions, list) and len(portions) == len(phases):
        for p, val in zip(phases, portions):
            p["portion"] = float(val)

    cache_base, output_base = derive_dirs(cfg)
    total_steps = int(cfg.get("total_steps", 1200))

    # Reparto de pasos por fase (garantizando ≥1 y que sumen total_steps).
    step_counts = [max(1, round(p["portion"] * total_steps)) for p in phases]
    step_counts[-1] += total_steps - sum(step_counts)

    PHASE_SETTINGS_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print(f"PROGRESSIVE TRAINING / ENTRENAMIENTO PROGRESIVO: {preset}")
    print(f"  Cache base   : {cache_base}")
    print(f"  Output base  : {output_base}")
    print(f"  Total steps  : {total_steps}")
    for i, (p, sc) in enumerate(zip(phases, step_counts)):
        print(f"  Fase {i}: {p['label']}²  {sc} pasos  gc={'ON' if p['gc'] else 'OFF'}")
    print("=" * 70, flush=True)

    prev_resume = None
    t0_all = time.time()

    for i, (phase, sc) in enumerate(zip(phases, step_counts)):
        if _stop_requested:
            print("\n[!] Stop requested; aborting remaining phases / Parada solicitada.")
            break

        label = phase["label"]
        phase_cache = os.path.join(cache_base, label)
        if not os.path.isdir(phase_cache):
            print(f"\n[!] Cache subdir not found for phase {i}: {phase_cache}")
            print("[!] ¿Ejecutaste el Pre-Caché en modo progresivo? / Did you run the progressive Pre-Cache?")
            return 1

        phase_output = os.path.join(output_base, f"phase{i}_{label}")

        # Settings de la fase: sin project_name para que cache/output explícitos manden.
        phase_cfg = dict(cfg)
        phase_cfg.pop("project_name", None)
        phase_cfg.pop("progressive", None)
        phase_cfg.pop("phase_portions", None)
        phase_cfg.update({
            "cache_dir": phase_cache,
            "output_dir": phase_output,
            "total_steps": int(sc),
            "gradient_checkpointing": bool(phase["gc"]),
            "init_lora_from": prev_resume or "",
        })
        if "batch" in phase:
            phase_cfg["batch_size"] = int(phase["batch"])

        phase_settings_path = PHASE_SETTINGS_DIR / f"phase{i}_{label}.json"
        with open(phase_settings_path, "w", encoding="utf-8") as f:
            json.dump(phase_cfg, f, indent=2, ensure_ascii=False)

        print(f"\n{'#' * 70}")
        print(f"# FASE {i+1}/{len(phases)} — {label}²  ({sc} pasos)")
        print(f"{'#' * 70}", flush=True)

        env = dict(os.environ)
        env["TRAIN_SETTINGS_PATH"] = str(phase_settings_path)

        global _current_child
        _current_child = subprocess.Popen(
            [sys.executable, "-u", str(TRAIN_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
        )
        ret = _current_child.wait()
        _current_child = None

        if ret != 0:
            print(f"\n[!] Phase {i} exited with code {ret} / La fase {i} terminó con código {ret}.")
            if _stop_requested:
                break
            print("[!] Aborting pipeline / Abortando el pipeline.")
            return ret

        prev_resume = os.path.join(phase_output, "resume_checkpoint")
        if not os.path.exists(os.path.join(prev_resume, "adapter_model.safetensors")):
            print(f"\n[!] No hand-off checkpoint after phase {i}: {prev_resume}")
            print("[!] No se puede continuar sin los pesos de la fase. / Cannot continue without phase weights.")
            return 1

    # Exportar el LoRA de la última fase completada al nombre final del usuario.
    if prev_resume and not _stop_requested:
        last_output = os.path.dirname(prev_resume)
        last_lora = os.path.join(last_output, "Krea2_FINAL_LoRA.safetensors")
        final_name = str(cfg.get("export_final_name", "")).strip() or "Krea2_Progressive_FINAL.safetensors"
        dest_dir = str(cfg.get("export_models_dir", "")).strip() or output_base
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, final_name)
        if os.path.exists(last_lora):
            try:
                shutil.copy2(last_lora, dest)
                print(f"\n✓ Final progressive LoRA / LoRA progresivo final: {dest}")
            except Exception as exc:
                print(f"[!] Could not copy final LoRA / No se pudo copiar el LoRA final: {exc}")

    dt = time.time() - t0_all
    print(f"\n✓ Progressive pipeline finished in / Pipeline progresivo terminado en "
          f"{int(dt//3600):02d}:{int((dt%3600)//60):02d}:{int(dt%60):02d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
