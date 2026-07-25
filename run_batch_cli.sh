#!/usr/bin/env bash

# AcademiaSD - Krea-2 LoRA Trainer CLI / Batch Runner for Linux
# Ejecuta pre-cache y/o entrenamiento sin necesidad de la interfaz web Flask.

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON=""

# Buscar entorno virtual
if [ -f "$BASE_DIR/.venv/bin/python" ]; then
    VENV_PYTHON="$BASE_DIR/.venv/bin/python"
elif [ -f "$BASE_DIR/venv/bin/python" ]; then
    VENV_PYTHON="$BASE_DIR/venv/bin/python"
elif [ -f "$BASE_DIR/env/bin/python" ]; then
    VENV_PYTHON="$BASE_DIR/env/bin/python"
fi

if [ -z "$VENV_PYTHON" ]; then
    echo "[ERROR] Entorno virtual no encontrado."
    echo "Ejecute ./install_LoRAlab-Krea2.sh primero."
    exit 1
fi

MODE="${1:-all}"  # Opciones: precache, train, all

echo "================================================================"
echo "   KREA-2 LORA TRAINER - PROCESO POR LOTE ALTERNO (CLI LINUX)"
echo "================================================================"
echo "Modo seleccionado: $MODE"
echo "Python: $VENV_PYTHON"
echo "================================================================"
echo ""

case "$MODE" in
    precache)
        echo "[1/1] Ejecutando Pre-Caché..."
        "$VENV_PYTHON" "$BASE_DIR/scripts/python/1_pre_cache_krea2.py"
        ;;
    train)
        echo "[1/1] Ejecutando Entrenamiento LoRA..."
        "$VENV_PYTHON" "$BASE_DIR/scripts/python/2_train_lora_krea2.py"
        ;;
    all)
        echo "[1/2] Paso 1: Ejecutando Pre-Caché..."
        "$VENV_PYTHON" "$BASE_DIR/scripts/python/1_pre_cache_krea2.py"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "[2/2] Paso 2: Ejecutando Entrenamiento LoRA..."
            "$VENV_PYTHON" "$BASE_DIR/scripts/python/2_train_lora_krea2.py"
        else
            echo "[ERROR] El Pre-Caché falló. Cancelando entrenamiento."
            exit 1
        fi
        ;;
    *)
        echo "Uso: $0 [precache|train|all]"
        echo "  precache : Ejecuta sólo 1_pre_cache_krea2.py"
        echo "  train    : Ejecuta sólo 2_train_lora_krea2.py"
        echo "  all      : Ejecuta ambos secuencialmente (por defecto)"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Proceso por lote finalizado."
echo "================================================================"
