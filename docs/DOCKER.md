# LoRAlab Docker Deployment Guide

Guía completa para construir, probar y desplegar LoRAlab en Docker (local y RunPod).

## Tabla de contenidos

1. [Requisitos](#requisitos)
2. [Testing Local](#testing-local)
3. [Build y Push a Registry](#build-y-push)
4. [Deployment en RunPod](#deployment-en-runpod)
5. [Seguridad](#seguridad)
6. [Troubleshooting](#troubleshooting)

---

## Requisitos

### Local (testing)
- **Docker Desktop** con WSL2 (Windows/Mac) o **Docker CLI** (Linux)
- **nvidia-docker** o **nvidia-container-runtime** (si tienes GPU NVIDIA)
- **Git** para clonar repo
- ~50 GB de espacio disco (modelo + cache + outputs)

### RunPod
- Cuenta en RunPod.io
- Créditos de GPU suficientes
- Conexión SSH desde tu máquina

---

## Testing Local

### 1. Preparar archivo .env

```bash
cd /path/to/AcademiaSD_LoRAlab-Krea2
cp .env.example .env
```

Editar `.env`:
```bash
HUGGINGFACE_TOKEN=hf_tu_token_aqui  # Obligatorio para descargar modelo
ADMIN_USER=admin
ADMIN_PASS=tu_password_seguro       # Cambiar esto
```

### 2. Crear directorios de volúmenes

```bash
mkdir -p {datasets,output,cache,models,ssh,certs,auth,logs}
chmod 755 {datasets,output,cache,models,ssh,certs,auth,logs}
```

### 3. Build image Docker

```bash
docker build -t loralab:latest .
# Tamaño esperado: ~5.5 GB
# Tiempo: 5-10 minutos (depende de conexión)
```

Verificar:
```bash
docker images | grep loralab
# loralab                    latest      <ID>   5.5GB
```

### 4. Lanzar con docker-compose

```bash
docker-compose up --detach
# o docker-compose up (foreground, ver logs)
```

Verificar servicios:
```bash
docker-compose ps
# NAME               STATUS
# loralab_trainer    Up X minutes
```

Ver logs:
```bash
docker-compose logs -f loralab
```

### 5. Test SSH

```bash
ssh -p 2222 loralab@localhost
# Prompt: password = valor de ADMIN_PASS en .env
```

Transferir dataset:
```bash
scp -P 2222 my_images.zip loralab@localhost:/data/datasets/
```

### 6. Test Web UI

Abrir navegador:
```
https://localhost:5000
```

Aceptar certificado autofirmado (warning normal).

Login con credenciales:
- Usuario: valor de `ADMIN_USER` en .env
- Contraseña: valor de `ADMIN_PASS` en .env

Si ves el dashboard: ✅ Funcionando

### 7. Test entrenamiento (opcional)

Desde Web UI:
1. Subir dataset vía SSH o drag-drop
2. Ejecutar Pre-cache
3. Ejecutar Train
4. Monitor logs en terminal

### 8. Detener

```bash
docker-compose down
# Preserve volumes: docker-compose down -v para limpiar
```

---

## Build y Push

### A Docker Hub

```bash
# Login
docker login

# Tag
docker tag loralab:latest username/loralab:latest
docker tag loralab:latest username/loralab:stable

# Push
docker push username/loralab:latest
docker push username/loralab:stable
```

### A AWS ECR

```bash
# Login a ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag loralab:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/loralab:latest

# Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/loralab:latest
```

### A GitHub Container Registry (GHCR)

```bash
# Login
echo $GH_TOKEN | docker login ghcr.io -u username --password-stdin

# Tag
docker tag loralab:latest ghcr.io/username/loralab:latest

# Push
docker push ghcr.io/username/loralab:latest
```

---

## Deployment en RunPod

### 1. Preparar imagen en registry público

Ver sección anterior (Docker Hub, ECR o GHCR). Asegurarse de que la imagen sea **pública** o tengas credenciales.

### 2. Crear Pod en RunPod

1. Login en https://runpod.io/console/pods
2. Click "Deploy" → "Custom Image"
3. **Container Image:** `username/loralab:latest` (o tu registry)
4. **GPU:** Seleccionar (A100, RTX 4090, etc.)
5. **Container Disk:** 100 GB mínimo (para modelo + cache)
6. **Volume:** Mount persistente para `/data` (recomendado 200 GB)

### 3. Configurar Variables de Entorno

En "Environment Variables" (RunPod):

```
HUGGINGFACE_TOKEN=hf_tu_token
ADMIN_USER=admin
ADMIN_PASS=super_secure_password
TRUSTED_IPS=<tu_IP_publica>
```

Obtener tu IP pública:
```bash
curl https://ifconfig.me
```

### 4. Ports

RunPod mapea puertos automáticamente. Verificar:
- **5000** → Web UI (HTTPS)
- **2222** → SSH (remapped desde 22)

RunPod te dará URLs públicas como:
- `https://abc123-5000.runpod.io`
- `ssh -p abc123-2222 loralab@runpod.io`

### 5. Test desde local

```bash
# Obtener hostname desde RunPod UI
RUNPOD_HOST=abc123.runpod.io

# SSH
ssh -p 2222 loralab@$RUNPOD_HOST

# SCP
scp -P 2222 dataset.zip loralab@$RUNPOD_HOST:/data/datasets/

# Web
curl -k https://$RUNPOD_HOST:5000/api/profile  # Requiere login
```

### 6. Primera ejecución

Primer boot tardará **3-5 minutos** (descargando modelo Krea-2).

Monitorear:
```bash
docker logs -f <container_id>  # Si acceso a host
# O desde Pod UI: ver logs
```

Cuando veas `[OK] Flask server iniciado`, estás listo.

---

## Seguridad

### Recomendaciones de Producción

#### 1. Cambiar contraseña por defecto INMEDIATAMENTE

```bash
ssh -p 2222 loralab@runpod_host
python /opt/loralab/scripts/python/auth_module.py
# O mejor: cambiar desde web UI (próxima versión)
```

#### 2. IP Whitelisting

Establecer en .env o RunPod env:
```
TRUSTED_IPS=203.0.113.42,198.51.100.0/24
```

Solo esas IPs pueden conectar.

#### 3. TLS Certificados Autofirmados

Los certificados se generan automáticamente con validez de 365 días.

Navegadores mostrarán advertencia (normal para autofirmados).

Para producción, considerar:
- Let's Encrypt (requiere dominio + cron renewal)
- Certificados corporativos

#### 4. SSH Hardening

Config incluida en `scripts/config/sshd_config`:
- ✅ PermitRootLogin: no
- ✅ MaxAuthTries: 3
- ✅ MaxSessions: 5
- ✅ PasswordAuthentication: yes
- ✅ AllowUsers: solo loralab

#### 5. Rate Limiting

Activo por defecto:
- Login: 5 intentos / 15 min
- Train/precache: 1 / 10 min
- Otros: 100 / min

#### 6. Firewall del host (RunPod)

Usar network policies de RunPod para:
- Bloquear IPs sospechosas
- Limitar ancho de banda
- Monitorear conexiones

### Auditoría y Logging

Los logs se guardan en `/data/logs/`:
```
training_logs/          # Logs de entrenamiento
ssh.log                 # SSH activity
flask.log               # Web server
```

Copiar regularmente para auditoría:
```bash
scp -rP 2222 loralab@host:/data/logs ./backup/
```

---

## Troubleshooting

### Problema: "Model download timeout"

**Causa:** Conexión lenta o token HF inválido.

**Solución:**
```bash
# Descargar manualmente en local
huggingface-cli download multimodalart/Krea-2-NF4 \
  --local-dir ./models/Krea-2-NF4 \
  --token hf_token

# Copiar a RunPod
scp -rP 2222 ./models/Krea-2-NF4 loralab@host:/data/models/

# Restart pod
```

### Problema: "SSH connection refused"

**Causa:** sshd no lanzado o puerto cerrado.

**Solución:**
```bash
# Ver logs
docker-compose logs loralab | grep SSH

# Verificar puerto
docker-compose exec loralab ss -tlnp | grep :22

# Reintentar
docker-compose restart loralab
```

### Problema: "GPU not detected in container"

**Causa:** nvidia-runtime no configurado.

**Solución:**
```bash
# Verificar en host
nvidia-smi

# Verificar en docker
docker run --rm --gpus all nvidia/cuda:12.4-runtime nvidia-smi

# Editar ~/.docker/daemon.json (Linux):
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Restart Docker daemon
sudo systemctl restart docker

# Retry docker-compose up
```

### Problema: "Web UI login falla"

**Causa:** Credenciales incorrectas o cookie expirada.

**Solución:**
```bash
# Verificar credenciales en container
docker-compose exec loralab cat /data/auth/credentials.json

# Reset credenciales
docker-compose exec loralab python /opt/loralab/scripts/python/init_auth.py

# Limpiar cookies en navegador (dev tools → Application → Cookies → delete)

# Reintentar login
```

### Problema: "Out of memory"

**Causa:** Batch size muy grande o modelo con precisión completa.

**Solución:**
```bash
# Editar train_settings.json en Web UI o vía SSH:
ssh loralab@host "cat /data/datasets/train_settings.json | jq .batch_size"

# Reducir batch_size o gradient_checkpointing
```

### Problema: "Certificate verification failed"

**Causa:** Navegador rechaza certificado autofirmado.

**Solución:**
```bash
# Opción 1: Aceptar excepción en navegador (click "Advanced" en warning)
# Opción 2: Instalar cert en sistema
# Opción 3: Usar curl con -k (ignore certs):
curl -k https://localhost:5000
```

---

## Actualizaciones

### Actualizar imagen

```bash
# Pull latest
git pull origin main

# Rebuild
docker build -t loralab:latest .

# Restart
docker-compose down
docker-compose up
```

Datos en `/data/*` se preservan (volumes).

---

## Soporte

Para problemas:
1. Revisar logs: `docker-compose logs -f`
2. SSH al container: `docker-compose exec loralab bash`
3. Verificar espacio: `docker-compose exec loralab df -h`
4. Reportar issue con logs en GitHub

---

**Última actualización:** 2026-07-26
