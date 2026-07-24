#!/usr/bin/env bash

# Triton and SageAttention Installer for Linux

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$BASE_DIR/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Entorno virtual no encontrado en $VENV_PYTHON"
    echo "Ejecute ./install_LoRAlab-Krea2.sh primero."
    exit 1
fi

echo "======================================================="
echo "   Instalación de Triton & SageAttention (Linux)"
echo "======================================================="

# En Linux Triton se instala directamente vía pip
echo "[INFO] Instalando Triton nativo de Linux..."
"$VENV_PYTHON" -m pip install -U triton

# Instalar SageAttention desde PyPI / fuente
echo "[INFO] Instalando SageAttention..."
"$VENV_PYTHON" -m pip install -U sageattention

echo "======================================================="
echo "   Proceso finalizado"
echo "======================================================="
