#!/usr/bin/env bash

# AcademiaSD - Krea-2 LoRA Trainer runner script for Linux

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE=""

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

if [ ! -f "$BASE_DIR/server.py" ]; then
    echo "[ERROR] No existe server.py"
    exit 1
fi

if [ ! -f "$BASE_DIR/trainer_ui.html" ]; then
    echo "[ERROR] No existe trainer_ui.html"
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

"$PYTHON_EXE" "$BASE_DIR/server.py"
