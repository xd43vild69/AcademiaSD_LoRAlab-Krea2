#!/bin/bash
# Generar certificados TLS autofirmados para LoRAlab
# Se ejecuta al iniciar el container si los certs no existen

CERT_DIR="/data/certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

# Crear directorio si no existe
mkdir -p "$CERT_DIR"

# Verificar si los certificados ya existen
if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "[OK] Certificados TLS encontrados en $CERT_DIR"
    exit 0
fi

echo "[*] Generando certificados TLS autofirmados..."

# Generar clave privada RSA 4096-bit + certificado X.509 válido por 365 días
openssl req -x509 \
    -newkey rsa:4096 \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -days 365 \
    -nodes \
    -subj "/CN=loralab.local/O=LoRAlab/ST=Localhost" \
    2>/dev/null

if [ $? -eq 0 ]; then
    # Establecer permisos seguros
    chmod 600 "$KEY_FILE"
    chmod 644 "$CERT_FILE"
    echo "[OK] Certificados generados:"
    echo "      Certificado: $CERT_FILE"
    echo "      Clave:       $KEY_FILE"
    echo "      Válido por:  365 días"
    exit 0
else
    echo "[!] Error generando certificados" >&2
    exit 1
fi
