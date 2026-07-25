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

# Elegir entrenador según 'progressive' en train_settings.json, igual que hace el
# botón Train de la UI. Sin esto, un proyecto progresivo lanzaba el entrenador
# directo contra el directorio padre de la caché (que sólo tiene subdirs 512/,
# 768/, 1024/), éste imprimía "caché vacía" y salía con código 0: un no-op
# silencioso que el script reportaba como éxito.
TRAIN_SCRIPT="$BASE_DIR/scripts/python/2_train_lora_krea2.py"
PROGRESSIVE="$("$VENV_PYTHON" -c "
import json, sys
try:
    with open('$BASE_DIR/train_settings.json', encoding='utf-8') as f:
        print(str(json.load(f).get('progressive', 'off')).strip().lower())
except Exception:
    print('off')
" 2>/dev/null)"

if [ -n "$PROGRESSIVE" ] && [ "$PROGRESSIVE" != "off" ] && [ "$PROGRESSIVE" != "none" ]; then
    TRAIN_SCRIPT="$BASE_DIR/scripts/python/run_progressive.py"
fi

echo "================================================================"
echo "   KREA-2 LORA TRAINER - PROCESO POR LOTE ALTERNO (CLI LINUX)"
echo "================================================================"
echo "Modo seleccionado: $MODE"
echo "Python: $VENV_PYTHON"
echo "Entrenador: $(basename "$TRAIN_SCRIPT") (progressive=$PROGRESSIVE)"
echo "================================================================"
echo ""

STATUS=0

case "$MODE" in
    precache)
        echo "[1/1] Ejecutando Pre-Caché..."
        "$VENV_PYTHON" "$BASE_DIR/scripts/python/1_pre_cache_krea2.py"
        STATUS=$?
        ;;
    train)
        echo "[1/1] Ejecutando Entrenamiento LoRA..."
        "$VENV_PYTHON" "$TRAIN_SCRIPT"
        STATUS=$?
        ;;
    all)
        echo "[1/2] Paso 1: Ejecutando Pre-Caché..."
        "$VENV_PYTHON" "$BASE_DIR/scripts/python/1_pre_cache_krea2.py"

        if [ $? -eq 0 ]; then
            echo ""
            echo "[2/2] Paso 2: Ejecutando Entrenamiento LoRA..."
            "$VENV_PYTHON" "$TRAIN_SCRIPT"
            STATUS=$?
        else
            echo "[ERROR] El Pre-Caché falló. Cancelando entrenamiento."
            exit 1
        fi
        ;;
    *)
        echo "Uso: $0 [precache|train|all]"
        echo "  precache : Ejecuta sólo 1_pre_cache_krea2.py"
        echo "  train    : Ejecuta sólo 2_train_lora_krea2.py (o run_progressive.py)"
        echo "  all      : Ejecuta ambos secuencialmente (por defecto)"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
if [ $STATUS -eq 0 ]; then
    echo "Proceso por lote finalizado."
else
    echo "[ERROR] Proceso por lote finalizado con código $STATUS."
fi
echo "================================================================"
exit $STATUS
