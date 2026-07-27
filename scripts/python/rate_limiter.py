"""Rate limiting y IP whitelisting para Flask.

Configura límites de requests y restricciones de IP para endpoints.
"""

import os
from functools import wraps
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def init_rate_limiter(app: Flask) -> Limiter:
    """Inicializar limiter con configuración de LoRAlab."""
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["100 per minute"],  # Default global
        storage_uri="memory://",  # In-memory (sin Redis)
    )
    return limiter


def apply_rate_limits(limiter: Limiter, app: Flask) -> None:
    """Aplicar límites específicos a endpoints de LoRAlab."""

    # POST /api/login: 5 intentos cada 15 minutos
    @app.route('/api/login', methods=['POST'])
    @limiter.limit("5 per 15 minutes")
    def login():
        # Implementado en server.py
        pass

    # POST /api/train: 1 inicio cada 10 minutos
    @app.route('/api/train', methods=['POST'])
    @limiter.limit("1 per 10 minutes")
    def train():
        pass

    # POST /api/precache: 1 cada 20 minutos
    @app.route('/api/precache', methods=['POST'])
    @limiter.limit("1 per 20 minutes")
    def precache():
        pass

    # GET /api/stream: unlimited (SSE long-lived)
    @app.route('/api/stream')
    @limiter.exempt
    def stream():
        pass

    # POST /api/curate: 1 cada 10 minutos
    @app.route('/api/curate', methods=['POST'])
    @limiter.limit("1 per 10 minutes")
    def curate():
        pass

    # GET /api/cancel: 5 per minute
    @app.route('/api/cancel', methods=['POST'])
    @limiter.limit("5 per minute")
    def cancel():
        pass

    # Otros endpoints API: 100 per minute (default)


def create_ip_whitelist_middleware(app: Flask) -> None:
    """Crear middleware que valida IPs whitelistadas."""

    def parse_trusted_ips():
        """Parsear TRUSTED_IPS del env (ej: '192.168.1.0/24,10.0.0.1')."""
        trusted = os.getenv("TRUSTED_IPS", "").strip()
        if not trusted:
            return None  # Sin whitelist, permite todo

        return [ip.strip() for ip in trusted.split(",")]

    trusted_ips = parse_trusted_ips()

    @app.before_request
    def check_ip_whitelist():
        """Rechazar si la IP no está en whitelist."""
        if not trusted_ips:
            return  # Sin whitelist activa

        client_ip = get_remote_address()

        # Permitir localhost siempre
        if client_ip in ("127.0.0.1", "::1", "localhost"):
            return

        # Permitir si está en whitelist (simple, sin CIDR)
        if client_ip in trusted_ips:
            return

        # Rechazar
        return jsonify({"error": "IP not whitelisted"}), 403


def get_rate_limiter_error_handler(app: Flask) -> None:
    """Registrar handler para errores de rate limiting."""
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({"error": "Too many requests", "message": str(e.description)}), 429
