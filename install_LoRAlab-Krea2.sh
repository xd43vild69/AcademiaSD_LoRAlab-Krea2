#!/usr/bin/env bash

# Instalador Venv Krea-2 Trainer para Linux (NVIDIA GPU)

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE=""

echo "========================================================"
echo "   INSTALADOR KREA-2 LORA TRAINER (LINUX)"
echo "   Entorno Python + PyTorch CUDA"
echo "========================================================"
echo ""

# 1. Comprobar Python 3 en el sistema
if command -v python3.13 &>/dev/null; then
    PYTHON_EXE="python3.13"
elif command -v python3.12 &>/dev/null; then
    PYTHON_EXE="python3.12"
elif [ -x "/usr/bin/python3" ]; then
    PYTHON_EXE="/usr/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
fi

if [ -n "$PYTHON_EXE" ]; then
    echo "[OK] Python detectado: $($PYTHON_EXE --version)"
else
    echo "[ERROR] Python 3 no está instalado en el sistema."
    echo "Por favor instálelo usando su gestor de paquetes (ej: sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

# 2. Comprobar módulo venv
$PYTHON_EXE -m venv --help &>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] El módulo venv de Python no está disponible."
    echo "En Ubuntu/Debian, instálelo con: sudo apt install python3-venv"
    exit 1
fi

# 3. Comprobar GPU NVIDIA
echo "[INFO] Comprobando drivers NVIDIA..."
if command -v nvidia-smi &>/dev/null; then
    echo "[OK] nvidia-smi detectado:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "[ADVERTENCIA] nvidia-smi no encontrado. Asegúrese de tener los controladores NVIDIA instalados."
fi

# 4. Crear venv limpio
VENV_DIR="$BASE_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Eliminando entorno virtual anterior..."
    rm -rf "$VENV_DIR"
fi

echo "[INFO] Creando nuevo entorno virtual..."
$PYTHON_EXE -m venv "$VENV_DIR"
if [ $? -ne 0 ]; then
    echo "[ERROR] No se pudo crear el entorno virtual."
    exit 1
fi

VENV_PYTHON="$VENV_DIR/bin/python"

# 5. Actualizar herramientas de pip
echo "[INFO] Actualizando pip, setuptools y wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

# 6. Instalar PyTorch con soporte CUDA
echo "[INFO] Instalando PyTorch con soporte CUDA..."
"$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || "$VENV_PYTHON" -m pip install torch torchvision torchaudio
"$VENV_PYTHON" -m pip install --upgrade torchvision

# 7. Instalar dependencias Krea-2
echo "[INFO] Instalando Diffusers, Transformers, PEFT, Accelerate, Safetensors y Hugging Face Hub..."
"$VENV_PYTHON" -m pip install diffusers transformers peft accelerate safetensors huggingface_hub

echo "[INFO] Instalando BitsAndBytes y utilidades..."
"$VENV_PYTHON" -m pip install bitsandbytes sentencepiece protobuf psutil

# 8. Verificación final
echo ""
echo "========================================================"
echo "   VERIFICACIÓN FINAL DE INSTALACIÓN"
echo "========================================================"
"$VENV_PYTHON" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA compilada:', torch.version.cuda); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA')"

echo ""
echo "[OK] Instalación completada. Ejecute ./run_LoRAlab-Krea2.sh para iniciar el entrenador."
