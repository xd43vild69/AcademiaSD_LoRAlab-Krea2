#!/usr/bin/env bash

# AcademiaSD - Krea-2 LoRA Trainer runner script for Linux

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE=""

# Auto-wrap en tmux: el server sobrevive a la caída de la sesión SSH y correr
# este script otra vez reengancha a la sesión existente (-A) en vez de duplicar.
# Desactivable con NO_TMUX=1.
if [ -z "$TMUX" ] && [ -z "$NO_TMUX" ] && command -v tmux &>/dev/null; then
    echo ""
    echo "[INFO] Servidor en sesión tmux 'loralab'."
    echo "       Desconectar sin parar: Ctrl+B y luego D"
    echo "       Volver a la sesión:    ./run_LoRAlab-Krea2.sh  (o: tmux attach -t loralab)"
    echo "       Sin tmux:              NO_TMUX=1 ./run_LoRAlab-Krea2.sh"
    echo ""
    exec tmux new-session -A -s loralab "NO_TMUX=1 '$BASE_DIR/run_LoRAlab-Krea2.sh'"
fi

echo ""
echo "================================================================"
echo "       ACADEMIASD - KREA-2 LORA TRAINER (Linux)"
echo "================================================================"
echo ""
echo "Carpeta del entrenador:"
echo "$BASE_DIR"
echo ""

# Buscar entorno virtual existente
if [ -f "$BASE_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$BASE_DIR/.venv/bin/python"
elif [ -f "$BASE_DIR/venv/bin/python" ]; then
    PYTHON_EXE="$BASE_DIR/venv/bin/python"
elif [ -f "$BASE_DIR/env/bin/python" ]; then
    PYTHON_EXE="$BASE_DIR/env/bin/python"
elif [ -f "$BASE_DIR/../venv/bin/python" ]; then
    PYTHON_EXE="$BASE_DIR/../venv/bin/python"
fi

if [ -z "$PYTHON_EXE" ]; then
    echo "[ERROR] No se ha encontrado el entorno virtual."
    echo "Ejecute ./install_LoRAlab-Krea2.sh primero para crearlo."
    exit 1
fi

echo "Entorno Python encontrado:"
echo "$PYTHON_EXE"
echo ""

if [ ! -f "$BASE_DIR/scripts/python/server.py" ]; then
    echo "[ERROR] No existe scripts/python/server.py"
    exit 1
fi

if [ ! -f "$BASE_DIR/web/trainer_ui.html" ]; then
    echo "[ERROR] No existe web/trainer_ui.html"
    exit 1
fi

echo "Comprobando Python..."
"$PYTHON_EXE" --version || { echo "[ERROR] No se puede ejecutar Python."; exit 1; }

echo ""
echo "================================================================"
echo "Iniciando servidor web..."
echo "================================================================"
echo ""
echo "Abre en el navegador:"
echo ""
echo "    http://127.0.0.1:5000"
echo ""
echo "Presiona Ctrl+C para detener el servidor."
echo ""

# Reduce la fragmentacion de VRAM, decisiva en GPUs de 12 GB a 768x768 o mas.
# server.py lanza los scripts con subprocess.Popen sin env=, asi que lo heredan.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_EXE" "$BASE_DIR/scripts/python/server.py"
