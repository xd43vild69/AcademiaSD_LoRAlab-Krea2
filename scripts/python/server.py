# -*- coding: utf-8 -*-
"""
server.py — Backend web para AcademiaSD Krea-2 Trainer
Web backend for AcademiaSD Krea-2 Trainer
"""

import codecs
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
import importlib.util
import signal
import shutil
import re
import string
import logging
import webbrowser
from pathlib import Path
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# =============================================================================
# DEPENDENCIAS / DEPENDENCIES
# =============================================================================

def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name

    if importlib.util.find_spec(import_name) is not None:
        return

    print()
    print("=" * 70)
    print(f"[INFO] Installing missing package / Instalando paquete: '{package_name}'...")
    print(f"[INFO] Python: {sys.executable}")
    print("=" * 70)
    print()

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Failed to install '{package_name}'. Exit code: {exc.returncode}\n")
        raise

    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(f"Package '{package_name}' installed but import failed.")

    print(f"[OK] '{package_name}' installed successfully / instalado correctamente.")


ensure_package("Flask", "flask")
ensure_package("psutil", "psutil")

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory
)

# Silenciar registros HTTP recurrentes de Werkzeug en consola
logging.getLogger('werkzeug').setLevel(logging.ERROR)


# =============================================================================
# CONFIGURACIÓN / CONFIGURATION
# =============================================================================

# BASE_DIR es scripts/python/ (donde vive este fichero); PROJECT_ROOT es la raíz
# del proyecto, contra la que se resuelven datos, configuración y recursos web.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
WEB_DIR = PROJECT_ROOT / "web"

UI_FILE = WEB_DIR / "trainer_ui.html"
LOGO_FILE = ASSETS_DIR / "logo.png" if (ASSETS_DIR / "logo.png").exists() else PROJECT_ROOT / "logo.png"

PRECACHE_CONFIG = PROJECT_ROOT / "pre_cache_settings.json"
TRAIN_CONFIG = PROJECT_ROOT / "train_settings.json"

CURATE_SCRIPT = BASE_DIR / "0_curate_dataset.py"
PRECACHE_SCRIPT = BASE_DIR / "1_pre_cache_krea2.py"
TRAIN_SCRIPT = BASE_DIR / "2_train_lora_krea2.py"
PROGRESSIVE_SCRIPT = BASE_DIR / "run_progressive.py"

# Curaduría: el informe lo escribe 0_curate_dataset.py y se regenera en cada
# scan; los ajustes manuales (umbral y reasignaciones) viven aparte para que
# volver a puntuar no se lleve por delante las decisiones del usuario.
CURATION_REPORT_NAME = "curation_report.json"
CURATION_OVERRIDES_NAME = "curation_overrides.json"

# Carpetas contenedoras: todo lo generado se agrupa aquí en vez de en la raíz.
CACHE_ROOT = "cached_data_local"
OUTPUT_ROOT = "output_local"

# Los procesos lanzados escriben su salida a un log en disco (no a una PIPE) y
# su identidad queda en RUN_STATE_FILE. Así el entrenamiento no depende de que
# el navegador —ni siquiera este server— sigan vivos: cualquier server que
# arranque después redescubre el run leyendo este archivo.
LOGS_DIR = PROJECT_ROOT / "logs"
RUN_STATE_FILE = LOGS_DIR / ".run_state.json"
KEEP_LOGS = 10

# Miniaturas del Dataset Inspector. El grid pinta recuadros de ~115 px, así que
# servir el original (1-3 MB por imagen) era el mayor coste de toda la app: un
# re-render movía decenas de MB. Se cachean en disco y se sirven con max_age
# largo; la invalidación va por la URL, que lleva el mtime del original.
THUMB_DIR = PROJECT_ROOT / ".thumb_cache"
THUMB_WIDTHS = (192, 384)          # 1x y 2x del recuadro más grande del grid
IMAGE_MAX_AGE = 31536000           # 1 año: seguro porque la URL está versionada

app = Flask(__name__)

# Caché en memoria del último Popen lanzado por ESTE server (sirve para reap y
# terminate de emergencia). La fuente de verdad del estado es RUN_STATE_FILE.
active_process = None
active_script = None
process_lock = threading.RLock()


# =============================================================================
# UTILIDADES / UTILS
# =============================================================================

def get_windows_drives():
    drives = []
    if os.name == "nt":
        try:
            from ctypes import windll
            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    drives.append(f"{letter}:\\")
                bitmask >>= 1
        except Exception:
            pass
    return drives


def read_json_file(path, default=None):
    if default is None:
        default = {}
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else default
    except Exception as exc:
        print(f"[ERROR] Could not read {path}: {exc}")
        return default


def write_json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temp_path, path)


def resolve_config_path(value, default):
    if value is None or str(value).strip() == "":
        value = default
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_train_output_dir():
    cfg = read_json_file(TRAIN_CONFIG, {})
    proj = cfg.get("project_name", "").strip()
    default_output = f"{OUTPUT_ROOT}/default"
    if proj:
        return resolve_config_path(f"{OUTPUT_ROOT}/{proj}", default_output)
    return resolve_config_path(cfg.get("output_dir"), default_output)


def get_dataset_dir():
    cfg = read_json_file(PRECACHE_CONFIG, {"dataset_path": "./dataset"})
    return resolve_config_path(cfg.get("dataset_path"), "./dataset")


def get_script_for_name(script_name):
    if script_name == "curate":
        return CURATE_SCRIPT
    if script_name == "precache":
        return PRECACHE_SCRIPT
    if script_name == "train":
        # En modo progresivo, el botón Train lanza el orquestador de fases en vez
        # del entrenador único. La decisión sale de la config guardada.
        cfg = read_json_file(TRAIN_CONFIG, {})
        preset = str(cfg.get("progressive", "off")).strip().lower()
        if preset not in ("off", "", "none"):
            return PROGRESSIVE_SCRIPT
        return TRAIN_SCRIPT
    return None


def resolve_curation_group(score, threshold, override):
    """Grupo efectivo de una imagen: "good" o "bad".

    Esta misma lógica está replicada en 2_train_lora_krea2.py; si cambia una,
    cambia la otra, o la UI enseñaría un reparto distinto del que se entrena.

    Reglas, en orden:
      1. Una reasignación manual gana siempre.
      2. Sin cara detectada (score None) -> grupo bueno. No puntuable no es lo
         mismo que mala: un plano de espalda o un perfil extremo son material
         legítimo, y penalizarlos por no ser medibles sería un error.
      3. Sin umbral (dataset demasiado pequeño para estadística) -> grupo bueno.
      4. score >= umbral -> bueno; por debajo -> bajo.
    """
    if override in ("good", "bad"):
        return override
    if score is None or threshold is None:
        return "good"
    return "good" if score >= threshold else "bad"


def load_curation(dataset_dir):
    """Informe de curaduría + ajustes manuales, ya fusionados.

    Devuelve (meta, por_imagen) donde por_imagen es {stem: {score, group, override}}.
    Si no se ha curado nunca, meta["available"] es False y por_imagen va vacío.
    """
    report = read_json_file(dataset_dir / CURATION_REPORT_NAME, {})
    overrides = read_json_file(dataset_dir / CURATION_OVERRIDES_NAME, {})

    images = report.get("images") or {}
    auto_threshold = report.get("auto_threshold")
    # El umbral manual sólo cuenta si es un número; None significa "usa el auto".
    manual = overrides.get("threshold")
    threshold = manual if isinstance(manual, (int, float)) else auto_threshold
    group_overrides = overrides.get("groups") or {}

    weights = report.get("weights") or {}
    meta = {
        "available": bool(images),
        "generated": report.get("generated"),
        "baselines": report.get("baselines") or [],
        "auto_threshold": auto_threshold,
        "threshold": threshold,
        "weight_good": float(weights.get("good", 1.0)),
        "weight_bad": float(weights.get("bad", 0.5)),
    }

    per_image = {}
    for stem, entry in images.items():
        score = (entry or {}).get("score")
        override = group_overrides.get(stem)
        per_image[stem] = {
            "score": score,
            "override": override if override in ("good", "bad") else None,
            "group": resolve_curation_group(score, threshold, override),
        }
    return meta, per_image


def _pid_matches(pid, argv):
    """¿Sigue vivo el PID y es realmente nuestro proceso (no un PID reciclado)?"""
    try:
        import psutil
        proc = psutil.Process(int(pid))
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if argv:
            cmdline = proc.cmdline()
            # El último argumento es la ruta del script: suficiente para
            # distinguir nuestro entrenador de un PID reciclado por otro proceso.
            return bool(cmdline) and cmdline[-1] == argv[-1]
        return True
    except Exception:
        return False


def get_active_run():
    """Run activo según RUN_STATE_FILE, o None.

    El disco es la fuente de verdad: un server recién reiniciado ve el
    entrenamiento que dejó corriendo su predecesor. Si el PID murió (o fue
    reciclado por otro proceso), limpia el state file de paso.
    """
    global active_process
    global active_script

    with process_lock:
        if active_process is not None:
            active_process.poll()  # reap del zombie si nuestro hijo ya terminó

        state = read_json_file(RUN_STATE_FILE, {})
        if not state.get("pid"):
            return None
        if not _pid_matches(state["pid"], state.get("argv")):
            try:
                RUN_STATE_FILE.unlink()
            except OSError:
                pass
            active_process = None
            active_script = None
            return None
        return state


def _prune_logs():
    try:
        logs = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in logs[KEEP_LOGS:]:
            stale.unlink()
    except OSError:
        pass


def get_status():
    run = get_active_run()
    if run is None:
        return {"running": False, "script": None, "pid": None, "log": None}
    return {
        "running": True,
        "script": run.get("script"),
        "pid": run.get("pid"),
        "log": run.get("log"),
        "started": run.get("started"),
    }


# =============================================================================
# REQUESTER NATIVO DE WINDOWS (POPUP SELECCIONAR CARPETA)
# =============================================================================

@app.route("/api/select-folder", methods=["POST"])
def select_folder_native():
    try:
        data = request.get_json(force=True) or {}
        initial_dir = data.get("initial_dir", str(PROJECT_ROOT)).strip()
        
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = str(PROJECT_ROOT)

        selected_path = None

        if os.name != "nt" and not os.environ.get("DISPLAY"):
            return jsonify({"status": "not_supported", "path": None, "reason": "No GUI display available on server (SSH/headless)"})

        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            chosen = filedialog.askdirectory(
                title="Select Folder / Seleccionar Carpeta",
                initialdir=initial_dir
            )
            root.destroy()
            if chosen:
                selected_path = str(Path(chosen).resolve())
        except Exception:
            pass

        if not selected_path and os.name == "nt":
            try:
                ps_cmd = (
                    '[System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null; '
                    '$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; '
                    f'$dialog.SelectedPath = "{initial_dir}"; '
                    'if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.SelectedPath }'
                )
                creation_flag = subprocess.CREATE_NO_WINDOW
                output = subprocess.check_output(["powershell", "-command", ps_cmd], text=True, errors="ignore", creationflags=creation_flag).strip()
                if output:
                    selected_path = str(Path(output).resolve())
            except Exception:
                pass

        if selected_path:
            return jsonify({"status": "ok", "path": selected_path})
        else:
            return jsonify({"status": "cancelled", "path": None})

    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# MONITOR VRAM, RAM & TEMPERATURA GPU DE HARDWARE
# =============================================================================

@app.route("/api/system-stats", methods=["GET"])
def get_system_stats():
    import psutil
    
    ram = psutil.virtual_memory()
    ram_info = {
        "total_gb": round(ram.total / (1024**3), 2),
        "used_gb": round(ram.used / (1024**3), 2),
        "percent": ram.percent
    }

    vram_info = {
        "total_gb": 0.0,
        "used_gb": 0.0,
        "percent": 0.0,
        "temp_c": 0,
        "gpu_name": "N/A"
    }

    try:
        creation_flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.used,name,temperature.gpu", "--format=csv,nounits,noheader"]
        output = subprocess.check_output(cmd, text=True, errors="ignore", creationflags=creation_flag).strip().splitlines()[0]
        parts = [p.strip() for p in output.split(",")]
        total_m = float(parts[0])
        used_m = float(parts[1])
        vram_info["gpu_name"] = parts[2]
        vram_info["temp_c"] = int(float(parts[3]))
        vram_info["total_gb"] = round(total_m / 1024.0, 2)
        vram_info["used_gb"] = round(used_m / 1024.0, 2)
        vram_info["percent"] = round((used_m / total_m) * 100, 1) if total_m > 0 else 0.0
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                vram_info["gpu_name"] = torch.cuda.get_device_name(0)
                free_b, total_b = torch.cuda.mem_get_info(0)
                used_b = total_b - free_b
                vram_info["total_gb"] = round(total_b / (1024**3), 2)
                vram_info["used_gb"] = round(used_b / (1024**3), 2)
                vram_info["percent"] = round((used_b / total_b) * 100, 1) if total_b > 0 else 0.0
        except Exception:
            pass

    return jsonify({"ram": ram_info, "vram": vram_info})


# =============================================================================
# EXPLORADOR DE DIRECTORIOS (WEB BACKUP)
# =============================================================================

@app.route("/api/browse-dir", methods=["GET"])
def browse_dir():
    requested_path = request.args.get("path", str(PROJECT_ROOT))
    try:
        target = Path(requested_path).resolve()
        if not target.exists() or not target.is_dir():
            target = PROJECT_ROOT
    except Exception:
        target = PROJECT_ROOT

    parent = str(target.parent) if target.parent != target else str(target)
    dirs = []
    try:
        for item in sorted(target.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                dirs.append({"name": item.name, "path": str(item)})
    except Exception:
        pass

    image_count = 0
    try:
        image_count = sum(1 for f in target.iterdir() if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    except Exception:
        pass

    return jsonify({
        "current_path": str(target),
        "parent_path": parent,
        "directories": dirs,
        "drives": get_windows_drives(),
        "image_count": image_count
    })


# =============================================================================
# EXPORTAR LORA A CARPETA MODELS
# =============================================================================

@app.route("/api/export-lora", methods=["POST"])
def export_lora():
    try:
        data = request.get_json(force=True) or {}
        target_dir_str = data.get("target_dir", "").strip()
        custom_name = data.get("final_name", "").strip()

        if not target_dir_str:
            return jsonify({"status": "error", "error": "Please select a target folder / Por favor selecciona una carpeta de destino."}), 400

        target_dir = Path(target_dir_str).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return jsonify({"status": "error", "error": f"Target folder does not exist / Carpeta de destino no existe: {target_dir}"}), 400

        if not custom_name:
            custom_name = "krea2_lora.safetensors"
        if not custom_name.lower().endswith(".safetensors"):
            custom_name += ".safetensors"

        output_dir = get_train_output_dir()
        if not output_dir.exists():
            return jsonify({"status": "error", "error": f"Output folder does not exist / Carpeta de salida no existe: {output_dir}"}), 404

        final_file = output_dir / "Krea2_FINAL_LoRA.safetensors"
        source_file = None

        if final_file.exists():
            source_file = final_file
        else:
            candidates = []
            for f in output_dir.glob("*.safetensors"):
                match = re.search(r"step_(\d+)\.safetensors$", f.name, re.IGNORECASE)
                if match:
                    candidates.append((int(match.group(1)), f))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                source_file = candidates[0][1]

        if not source_file or not source_file.exists():
            return jsonify({"status": "error", "error": f"No .safetensors files found in / No se encontraron archivos .safetensors en: {output_dir}"}), 404

        dest_file = target_dir / custom_name
        shutil.copy2(source_file, dest_file)

        return jsonify({
            "status": "ok",
            "source": source_file.name,
            "dest": str(dest_file),
            "filename": custom_name
        })

    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# UI, ASSETS Y LOGO (FAVICON)
# =============================================================================

@app.route("/")
def index():
    if not UI_FILE.exists():
        return f"File not found / No se encuentra: trainer_ui.html in {WEB_DIR}", 404
    return send_from_directory(str(WEB_DIR), UI_FILE.name)


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    if ASSETS_DIR.exists():
        return send_from_directory(str(ASSETS_DIR), filename)
    return send_from_directory(str(PROJECT_ROOT), filename)


@app.route("/favicon.ico")
@app.route("/logo.png")
def serve_logo():
    if (ASSETS_DIR / "logo.png").exists():
        return send_from_directory(str(ASSETS_DIR), "logo.png")
    if (PROJECT_ROOT / "logo.png").exists():
        return send_from_directory(str(PROJECT_ROOT), "logo.png")
    return "", 404


# =============================================================================
# CONFIGURACIÓN JSON (GUARDADO EN 2 SITIOS)
# =============================================================================

@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify({
        "pre_cache": read_json_file(PRECACHE_CONFIG, {}),
        "train": read_json_file(TRAIN_CONFIG, {}),
        "base_dir": str(PROJECT_ROOT)
    })


@app.route("/api/save-precache", methods=["POST"])
def save_precache():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "error": "JSON object required / Objeto JSON requerido."}), 400

        # Merge, no reemplazo: la UI sólo envía las claves que sabe pintar, así que un
        # write directo borraba en silencio todo lo demás que el script sí lee.
        data = {**read_json_file(PRECACHE_CONFIG, {}), **data}

        proj = data.get("project_name", "").strip()
        cache_dir_name = f"{CACHE_ROOT}/{proj}" if proj else f"{CACHE_ROOT}/default"
        output_dir_name = f"{OUTPUT_ROOT}/{proj}" if proj else f"{OUTPUT_ROOT}/default"

        data["cache_dir"] = f"./{cache_dir_name}"

        # 1. Guardar en raíz con nombre genérico pre_cache_settings.json (lectura por defecto)
        write_json_file(PRECACHE_CONFIG, data)
        saved_files = [PRECACHE_CONFIG.name]

        # 2. Guardar copia dentro de la carpeta del proyecto
        if proj:
            cache_dir_path = resolve_config_path(data["cache_dir"], cache_dir_name)
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            cache_json_file = cache_dir_path / f"pre_cache_settings_{proj}.json"
            write_json_file(cache_json_file, data)
            saved_files.append(f"{cache_dir_name}/{cache_json_file.name}")

        # Sincronizar train_settings.json en raíz
        train_cfg = read_json_file(TRAIN_CONFIG, {})
        if "dataset_path" in data:
            train_cfg["dataset_path"] = data["dataset_path"]
        if "project_name" in data:
            train_cfg["project_name"] = data["project_name"]
            train_cfg["cache_dir"] = data["cache_dir"]
            train_cfg["output_dir"] = f"./{output_dir_name}"
        if "trigger_word" in data:
            train_cfg["trigger_word"] = data["trigger_word"]
        # Mantener coherente el modo de resolución: el botón Train lo lee de
        # train_settings. Con una sola resolución en la lista, el pre-cache sigue
        # escribiendo en un subdir <label>/, así que el modo NO puede ser "off"
        # (el entrenador miraría el directorio padre y lo encontraría vacío):
        # se usa el preset de resolución constante con ese mismo label.
        res = data.get("resolutions")
        if isinstance(res, list):
            if len(res) == 1:
                train_cfg["progressive"] = str(int(round(math.sqrt(float(res[0])))))
            else:
                train_cfg["progressive"] = ("512_768_1024" if len(res) == 3
                                            else "768_1024" if len(res) == 2
                                            else "off")

        write_json_file(TRAIN_CONFIG, train_cfg)

        return jsonify({"status": "ok", "file": saved_files[0], "all_saved": saved_files})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/save-train", methods=["POST"])
def save_train():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "error": "JSON object required / Objeto JSON requerido."}), 400

        # Merge, no reemplazo: la UI construye un objeto de claves fijas, así que un write
        # directo borraba warmup_steps, min_lr_ratio, weight_decay, max_grad_norm,
        # timestep_sampling, model_id, gradient_checkpointing, preview_steps/cfg y hf_token.
        data = {**read_json_file(TRAIN_CONFIG, {}), **data}

        proj = data.get("project_name", "").strip()
        cache_dir_name = f"{CACHE_ROOT}/{proj}" if proj else f"{CACHE_ROOT}/default"
        output_dir_name = f"{OUTPUT_ROOT}/{proj}" if proj else f"{OUTPUT_ROOT}/default"

        data["cache_dir"] = f"./{cache_dir_name}"
        data["output_dir"] = f"./{output_dir_name}"

        # 1. Guardar en raíz con nombre genérico
        write_json_file(TRAIN_CONFIG, data)
        saved_files = [TRAIN_CONFIG.name]

        # 2. Guardar copia dentro de la carpeta del proyecto
        if proj:
            output_dir_path = resolve_config_path(data["output_dir"], output_dir_name)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_json_file = output_dir_path / f"train_settings_{proj}.json"
            write_json_file(output_json_file, data)
            saved_files.append(f"{output_dir_name}/{output_json_file.name}")

        return jsonify({"status": "ok", "file": saved_files[0], "all_saved": saved_files})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# EJECUCIÓN DE SCRIPT & STREAMING
# =============================================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(get_status())


@app.route("/api/checkpoint-info", methods=["GET"])
def checkpoint_info():
    output_dir = get_train_output_dir()
    step_file = output_dir / "current_step.txt"
    resume_dir = output_dir / "resume_checkpoint"
    
    has_checkpoint = step_file.exists() and resume_dir.exists()
    current_step = 0
    if step_file.exists():
        try:
            current_step = int(step_file.read_text(encoding="utf-8").strip())
        except Exception:
            pass
            
    return jsonify({
        "has_checkpoint": has_checkpoint,
        "current_step": current_step,
        "output_dir": str(output_dir)
    })


@app.route("/api/run", methods=["POST"])
def run_script():
    global active_process
    global active_script

    try:
        data = request.get_json(force=True) or {}
        script_name = data.get("script")
        script_path = get_script_for_name(script_name)

        if script_path is None or not script_path.exists():
            return jsonify({"status": "error", "error": f"Script not found / Script no encontrado: {script_name}"}), 404

        with process_lock:
            running = get_active_run()
            if running is not None:
                return jsonify({"status": "error", "error": f"Process already running / Proceso en ejecución: {running.get('script')}"}), 409

            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            _prune_logs()
            log_name = f"{script_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
            log_path = LOGS_DIR / log_name

            command = [sys.executable, "-u", str(script_path)]

            # El proceso vive en su propia sesión (POSIX) / grupo (Windows) y
            # escribe directo al log: sobrevive a la desconexión del navegador,
            # a la caída del SSH y a la muerte de este server.
            popen_kwargs = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            log_fh = open(log_path, "wb", buffering=0)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    **popen_kwargs,
                )
            finally:
                log_fh.close()  # el hijo conserva su propio descriptor

            write_json_file(RUN_STATE_FILE, {
                "pid": process.pid,
                "script": script_name,
                "log": log_name,
                "started": datetime.now().isoformat(timespec="seconds"),
                "argv": command,
            })
            active_process = process
            active_script = script_name

        return jsonify({"status": "started", "script": script_name, "pid": process.pid, "log": log_name})

    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# Separa conservando el terminador: split devuelve [texto, sep, texto, sep, …,
# resto], que es justo lo que hace falta para saber si la línea la cerró un
# '\r' (reescribir en sitio) o un '\n' (línea nueva).
LINE_SPLIT_RE = re.compile(r"([\r\n])")


def sse_output(text, replace, offset):
    payload = {"type": "output", "text": text, "replace": replace, "offset": offset}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.route("/api/stream", methods=["GET"])
def stream_log():
    """Tail SSE de un log de run. Reengancha sin tocar el proceso.

    ?log=<nombre>   log a seguir (de /api/run o /api/status)
    ?offset=<byte>  posición inicial; sin él, los últimos 64 KB
    """
    log_name = request.args.get("log", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.log", log_name):
        return jsonify({"status": "error", "error": "Invalid log name"}), 400
    log_path = LOGS_DIR / log_name

    try:
        start_offset = int(request.args.get("offset", "-1"))
    except ValueError:
        start_offset = -1

    def generate():
        pos = start_offset
        if pos < 0:
            try:
                pos = max(0, log_path.stat().st_size - 64 * 1024)
            except OSError:
                pos = 0

        yield f"data: {json.dumps({'type': 'start', 'log': log_name, 'offset': pos}, ensure_ascii=False)}\n\n"

        buffer = ""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = b""
                try:
                    with log_path.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read(65536)
                except OSError:
                    pass

                if chunk:
                    pos += len(chunk)
                    # El decodificador incremental mantiene el estado entre
                    # bloques: un carácter multibyte partido por el corte de los
                    # 64 KB ya no se corrompe.
                    buffer += decoder.decode(chunk)
                    # Trocear de una pasada en C en vez de iterar carácter a
                    # carácter en Python (65 536 vueltas y otras tantas
                    # concatenaciones por bloque, todas reteniendo el GIL y
                    # frenando el resto del servidor).
                    parts = LINE_SPLIT_RE.split(buffer)
                    buffer = parts[-1]          # resto sin terminador
                    for i in range(0, len(parts) - 1, 2):
                        text, sep = parts[i], parts[i + 1]
                        if text:
                            # '\r' reescribe la línea en sitio (barras de
                            # progreso); '\n' abre línea nueva.
                            yield sse_output(text, sep == "\r", pos)
                    continue  # puede haber más pendiente: no dormir aún

                run = get_active_run()
                if run is None or run.get("log") != log_name:
                    # El proceso terminó (o este log es de un run anterior) y no
                    # quedan bytes: cerrar con 'done'.
                    if buffer:
                        yield sse_output(buffer, False, pos)
                        buffer = ""
                    with process_lock:
                        code = active_process.returncode if active_process is not None else None
                    yield f"data: {json.dumps({'type': 'done', 'log': log_name, 'code': code, 'offset': pos}, ensure_ascii=False)}\n\n"
                    break

                time.sleep(0.3)
        except GeneratorExit:
            # El cliente se fue. El proceso escribe a disco y ni se entera.
            pass

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/stop", methods=["POST"])
def stop_script():
    run = get_active_run()
    if run is None:
        return jsonify({"status": "not_running"})

    pid = int(run["pid"])
    script = run.get("script")
    try:
        if os.name == "nt":
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            # SIGINT al PID (no al grupo): en modo progresivo el orquestador ya
            # reenvía la señal a su hijo. El entrenador guarda checkpoint y sale.
            os.kill(pid, signal.SIGINT)
        return jsonify({"status": "terminating", "script": script})
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        return jsonify({"status": "terminated", "script": script})


# =============================================================================
# PREVIEWS & DATASET API
# =============================================================================

@app.route("/api/previews", methods=["GET"])
def get_previews():
    output_dir = get_train_output_dir()
    previews = []
    if output_dir.is_dir():
        # scandir da el tipo y el stat de una pasada; el filtro por nombre va
        # antes de tocar el disco, así que los .safetensors del directorio ya no
        # se statean (este endpoint se llama cada 8 s).
        with os.scandir(output_dir) as it:
            for entry in it:
                if (entry.name.startswith("preview_step_")
                        and entry.name.lower().endswith(".png") and entry.is_file()):
                    previews.append((entry.stat().st_mtime, entry.name))
    previews.sort(reverse=True)
    return jsonify({"output_dir": str(output_dir), "previews": [n for _, n in previews[:50]]})


@app.route("/api/preview/<path:filename>")
def serve_preview(filename):
    output_dir = get_train_output_dir()
    requested = (output_dir / filename).resolve()
    try:
        requested.relative_to(output_dir.resolve())
    except ValueError:
        return "", 403
    if not requested.is_file():
        return "", 404
    # Los previews son inmutables: el nombre lleva el step. La galería los pedía
    # de nuevo cada 8 s porque Werkzeug fuerza no-cache sin max_age.
    return send_from_directory(str(output_dir), requested.name, max_age=IMAGE_MAX_AGE)


@app.route("/api/dataset-info", methods=["GET"])
def dataset_info():
    dataset_dir = get_dataset_dir()
    # La curaduría se adjunta aquí en vez de en su propio endpoint para que la
    # vista de dos grupos salga de una sola llamada, con los captions incluidos.
    curation_meta, curation_images = load_curation(dataset_dir)
    images = []
    if dataset_dir.is_dir():
        for file_path in sorted(dataset_dir.iterdir()):
            if file_path.is_file() and not file_path.name.startswith(".") and file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                txt_path = file_path.with_suffix(".txt")
                caption = ""
                # Un solo stat: leer el .txt ya nos dice si existe. Antes se
                # llamaba a exists() dos veces por imagen (2N syscalls por
                # request, y este endpoint se dispara desde 10 sitios).
                try:
                    caption = txt_path.read_text(encoding="utf-8").strip()
                    has_txt = True
                except FileNotFoundError:
                    has_txt = False
                except Exception:
                    # El .txt está pero no se puede leer (permisos, encoding):
                    # existe igualmente, sólo que sin caption utilizable.
                    has_txt = True
                try:
                    mtime = int(file_path.stat().st_mtime)
                except OSError:
                    mtime = 0
                # mtime viaja al cliente para versionar la URL de la imagen: así
                # se puede servir con max_age largo sin servir nada rancio.
                entry = {"file": file_path.name, "has_txt": has_txt,
                         "caption": caption, "mtime": mtime}
                cur = curation_images.get(file_path.stem)
                if cur is not None:
                    entry.update(score=cur["score"], group=cur["group"], override=cur["override"])
                else:
                    # Imagen añadida después del último scan: sin puntuación todavía.
                    entry.update(score=None, group=None, override=None)
                images.append(entry)
    return jsonify({
        "path": str(dataset_dir),
        "image_count": len(images),
        "caption_count": sum(1 for img in images if img["has_txt"]),
        "curation": curation_meta,
        "images": images[:500]
    })


@app.route("/api/curation-overrides", methods=["POST"])
def save_curation_overrides():
    """Guarda el umbral manual y las reasignaciones de grupo.

    Va a un archivo aparte del informe a propósito: 0_curate_dataset.py regenera
    curation_report.json en cada scan, y las decisiones del usuario no deben
    perderse por volver a puntuar.
    """
    try:
        data = request.get_json(force=True) or {}
        dataset_dir = get_dataset_dir()
        if not dataset_dir.is_dir():
            return jsonify({"status": "error", "error": f"Dataset folder not found / Carpeta no encontrada: {dataset_dir}"}), 404

        threshold = data.get("threshold")
        if threshold is not None and not isinstance(threshold, (int, float)):
            return jsonify({"status": "error", "error": "threshold must be a number or null / debe ser número o null."}), 400

        # Sólo se persiste lo que difiere del veredicto automático: guardar el
        # grupo de cada imagen convertiría un cambio de umbral posterior en un
        # no-op silencioso, porque todo estaría ya fijado a mano.
        groups = data.get("groups") or {}
        if not isinstance(groups, dict):
            return jsonify({"status": "error", "error": "groups must be an object / debe ser un objeto."}), 400
        clean_groups = {str(k): v for k, v in groups.items() if v in ("good", "bad")}

        payload = {"threshold": threshold, "groups": clean_groups}
        write_json_file(dataset_dir / CURATION_OVERRIDES_NAME, payload)

        meta, per_image = load_curation(dataset_dir)
        good = sum(1 for v in per_image.values() if v["group"] == "good")
        return jsonify({
            "status": "ok",
            "file": CURATION_OVERRIDES_NAME,
            "good_count": good,
            "bad_count": len(per_image) - good,
            "override_count": len(clean_groups),
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


def get_thumbnail(src: Path, width: int):
    """Ruta de la miniatura de `src` a `width` px, generándola si hace falta.

    Devuelve None si Pillow no está o la imagen no se puede procesar: quien
    llama sirve entonces el original, así que esto nunca deja al grid sin fotos.
    """
    try:
        stat = src.stat()
        # La clave incluye mtime y tamaño: tocar la imagen invalida la miniatura.
        key = hashlib.sha1(
            f"{src.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{width}".encode()
        ).hexdigest()
        thumb = THUMB_DIR / f"{key}.jpg"
        if thumb.exists():
            return thumb

        from PIL import Image
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((width, width), Image.LANCZOS)
            tmp = thumb.with_suffix(".tmp.jpg")
            im.save(tmp, "JPEG", quality=82, optimize=True)
            os.replace(tmp, thumb)
        return thumb
    except Exception:
        return None


@app.route("/api/dataset-image/<path:filename>")
def serve_dataset_image(filename):
    dataset_dir = get_dataset_dir()
    try:
        requested = (dataset_dir / filename).resolve()
        requested.relative_to(dataset_dir.resolve())
    except Exception:
        return "", 403
    if not requested.is_file() or requested.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        return "", 404

    # ?w=<px> pide miniatura (grid); sin él se sirve el original (modal/zoom).
    try:
        width = int(request.args.get("w", "0"))
    except ValueError:
        width = 0
    if width in THUMB_WIDTHS:
        thumb = get_thumbnail(requested, width)
        if thumb is not None:
            return send_from_directory(str(thumb.parent), thumb.name, max_age=IMAGE_MAX_AGE)

    return send_from_directory(str(dataset_dir), requested.name, max_age=IMAGE_MAX_AGE)


@app.route("/api/save-caption", methods=["POST"])
def save_caption():
    try:
        data = request.get_json(force=True)
        filename = data.get("filename")
        caption = data.get("caption", "").strip()
        dataset_dir = get_dataset_dir()
        
        img_path = (dataset_dir / filename).resolve()
        img_path.relative_to(dataset_dir.resolve())
        
        txt_path = img_path.with_suffix(".txt")
        txt_path.write_text(caption, encoding="utf-8")
        
        return jsonify({"status": "ok", "file": txt_path.name})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/delete-dataset-image", methods=["POST"])
def delete_dataset_image():
    try:
        data = request.get_json(force=True)
        filename = data.get("filename")
        if not filename:
            return jsonify({"status": "error", "error": "Filename is required"}), 400

        dataset_dir = get_dataset_dir()
        img_path = (dataset_dir / filename).resolve()
        img_path.relative_to(dataset_dir.resolve())

        if not img_path.is_file() or img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            return jsonify({"status": "error", "error": "File not found or invalid image format"}), 404

        # Delete the image file
        img_path.unlink()

        # Delete corresponding .txt caption file if it exists
        txt_path = img_path.with_suffix(".txt")
        if txt_path.is_file():
            try:
                txt_path.unlink()
            except Exception:
                pass

        return jsonify({"status": "ok", "deleted": filename})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500




@app.route("/api/batch-caption", methods=["POST"])
def batch_caption():
    try:
        data = request.get_json(force=True)
        trigger = data.get("trigger_word", "").strip()
        dataset_dir = get_dataset_dir()
        count = 0
        if trigger and dataset_dir.is_dir():
            for file_path in dataset_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    txt_path = file_path.with_suffix(".txt")
                    text = ""
                    if txt_path.exists():
                        text = txt_path.read_text(encoding="utf-8").strip()
                    if trigger.lower() not in text.lower():
                        text = f"{trigger}, {text}".strip(", ")
                        txt_path.write_text(text, encoding="utf-8")
                        count += 1
        return jsonify({"status": "ok", "updated_count": count})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/batch-replace-caption", methods=["POST"])
def batch_replace_caption():
    try:
        data = request.get_json(force=True)
        search_text = data.get("search_text", "")
        replace_text = data.get("replace_text", "")
        case_sensitive = bool(data.get("case_sensitive", True))
        whole_word = bool(data.get("whole_word", True))
        clean_commas = bool(data.get("clean_commas", True))
        
        if not search_text:
            return jsonify({"status": "error", "error": "Search text is required / Se requiere texto a buscar."}), 400

        dataset_dir = get_dataset_dir()
        total_occurrences = 0
        files_updated = 0

        if dataset_dir.is_dir():
            escaped_search = re.escape(search_text)
            if whole_word:
                pattern_str = r'(?<![a-zA-Z0-9_])' + escaped_search + r'(?![a-zA-Z0-9_])'
            else:
                pattern_str = escaped_search

            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern_str, flags)

            for file_path in dataset_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    txt_path = file_path.with_suffix(".txt")
                    if txt_path.exists():
                        try:
                            original_text = txt_path.read_text(encoding="utf-8")
                        except Exception:
                            continue

                        matches = len(pattern.findall(original_text))
                        if matches > 0:
                            total_occurrences += matches
                            new_text = pattern.sub(replace_text, original_text)

                            if clean_commas:
                                new_text = re.sub(r'\s*,\s*,+', ',', new_text)
                                new_text = re.sub(r'\s+,', ',', new_text)
                                new_text = re.sub(r'\s*,\s*', ', ', new_text)
                                new_text = re.sub(r'  +', ' ', new_text)
                                new_text = new_text.strip(', ')

                            txt_path.write_text(new_text, encoding="utf-8")
                            files_updated += 1

        return jsonify({
            "status": "ok",
            "occurrences_replaced": total_occurrences,
            "files_updated": files_updated
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


def open_browser():
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ACADEMIASD — KREA-2 LORA TRAINER WEB SERVER")
    print("=" * 70)
    print(f"  Base Dir / Carpeta  : {PROJECT_ROOT}")
    print(f"  Python Interpreter  : {sys.executable}")
    print(f"  URL                 : http://127.0.0.1:5000")
    print("=" * 70 + "\n")

    threading.Timer(1.2, open_browser).start()

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)