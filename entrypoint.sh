#!/bin/bash
# Entrypoint: inicializa servicios en el contenedor Docker
set -e

echo "[*] LoRAlab Docker Entrypoint Starting..."

# 1. Descargar modelo si no existe
if [ ! -d "/data/models/Krea-2-NF4" ]; then
    echo "[*] Descargando modelo Krea-2-NF4 desde HuggingFace..."
    python /opt/loralab/scripts/python/download_model.py || {
        echo "[!] Error descargando modelo. Continuando (modo CPU)..."
    }
else
    echo "[OK] Modelo Krea-2-NF4 encontrado en /data/models"
fi

# 2. Generar certificados TLS si no existen
echo "[*] Inicializando certificados TLS..."
/opt/loralab/scripts/shell/generate_certs.sh

# 3. Inicializar credenciales de login si no existen
if [ ! -f "/data/auth/credentials.json" ]; then
    echo "[*] Generando credenciales por defecto..."
    python /opt/loralab/scripts/python/init_auth.py
else
    echo "[OK] Credenciales encontradas"
fi

# 4. Crear directorios si no existen
mkdir -p /data/{datasets,output,cache,logs,ssh,certs,auth}

# 5. Inicializar SSH si es necesario
if [ ! -f "/etc/ssh/ssh_host_rsa_key" ]; then
    echo "[*] Generando claves SSH del host..."
    ssh-keygen -A
fi

# 6. Configurar permisos SSH
chmod 700 /data/ssh 2>/dev/null || true
chmod 600 /data/ssh/authorized_keys 2>/dev/null || true

# 7. Mostrar información de inicio
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  LoRAlab Training Platform (Dockerized)"
echo "════════════════════════════════════════════════════════════"
echo "  Admin User:     ${ADMIN_USER:-admin}"
echo "  Web UI:         https://localhost:${WEB_PORT:-5000}"
echo "  SSH:            ssh -p ${SSH_PORT:-22} loralab@localhost"
echo "  Datasets:       /data/datasets"
echo "  Output:         /data/output"
echo "  Cache:          /data/cache"
echo "════════════════════════════════════════════════════════════"
echo ""

# 8. Lanzar servicios en paralelo
echo "[*] Iniciando SSH daemon..."
/usr/sbin/sshd -D &
SSH_PID=$!
echo "[OK] SSH daemon iniciado (PID: $SSH_PID)"

sleep 1

echo "[*] Iniciando Flask web server..."
cd /opt/loralab
python scripts/python/server.py &
FLASK_PID=$!
echo "[OK] Flask server iniciado (PID: $FLASK_PID)"

echo "[*] Entrypoint completado. Servicios en ejecución."
echo ""

# 9. Capturar señales y apagar gracefully
trap "echo '[*] Recibida señal SIGTERM/SIGINT'; kill $SSH_PID $FLASK_PID 2>/dev/null || true; exit 0" SIGTERM SIGINT

# Esperar a que cualquiera de los dos procesos termine
wait -n

# Si uno muere, matar ambos
echo "[!] Un servicio ha terminado. Apagando el sistema..."
kill $SSH_PID $FLASK_PID 2>/dev/null || true
wait

exit 0
