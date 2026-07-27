"""Simple file-based authentication module for LoRAlab web UI.

Gestiona credenciales en JSON, sesiones en memoria, y validación de tokens.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from secrets import token_hex


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_CREDS_PATH = PROJECT_ROOT / "credentials.json"
CREDS_FILE = Path(os.getenv("AUTH_CREDS_FILE", str(DEFAULT_CREDS_PATH)))
SESSIONS = {}  # {token: {user, exp_time}}


def _hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hash password con salt."""
    if salt is None:
        salt = token_hex(16)
    hash_val = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    return hash_val, salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verificar contraseña contra hash almacenado."""
    computed_hash, _ = _hash_password(password, salt)
    return computed_hash == stored_hash


def load_users() -> dict:
    """Cargar usuarios desde JSON. Si no existe, retornar vacío."""
    if not CREDS_FILE.exists():
        return {}
    try:
        with open(CREDS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_users(users: dict) -> None:
    """Guardar usuarios a JSON de forma atómica."""
    CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = CREDS_FILE.with_suffix(".tmp")
    with open(tmp_file, "w") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp_file, CREDS_FILE)


def add_user(username: str, password: str, role: str = "user", overwrite: bool = True) -> bool:
    """Agregar o actualizar usuario."""
    if not username or not password:
        return False
    users = load_users()
    if username in users and not overwrite:
        return False  # Usuario ya existe y no se permite sobrescribir

    hash_val, salt = _hash_password(password)
    users[username] = {
        "hash": hash_val,
        "salt": salt,
        "role": role,
        "created_at": datetime.utcnow().isoformat()
    }
    save_users(users)
    return True


def authenticate(username: str, password: str) -> bool:
    """Validar credenciales."""
    users = load_users()
    if username not in users:
        return False

    user = users[username]
    return _verify_password(password, user["hash"], user["salt"])


def create_session_token(username: str, expiry_minutes: int = 60) -> str:
    """Crear token de sesión válido por N minutos."""
    token = token_hex(32)
    exp_time = datetime.utcnow() + timedelta(minutes=expiry_minutes)
    SESSIONS[token] = {
        "user": username,
        "exp_time": exp_time,
        "created_at": datetime.utcnow()
    }
    return token


def verify_session_token(token: str) -> str:
    """Verificar token válido, retorna username o None."""
    if token not in SESSIONS:
        return None

    session = SESSIONS[token]
    if datetime.utcnow() > session["exp_time"]:
        del SESSIONS[token]
        return None

    return session["user"]


def revoke_session(token: str) -> None:
    """Revocar sesión."""
    SESSIONS.pop(token, None)


def get_active_sessions() -> dict:
    """Retornar diccionario de sesiones activas."""
    now = datetime.utcnow()
    # Limpiar expiradas
    expired = [t for t, s in SESSIONS.items() if now > s["exp_time"]]
    for t in expired:
        del SESSIONS[t]
    return SESSIONS.copy()


def user_exists(username: str) -> bool:
    """Verificar si usuario existe."""
    return username in load_users()


def get_user_role(username: str) -> str:
    """Obtener rol del usuario."""
    users = load_users()
    return users.get(username, {}).get("role", "user")


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Cambiar contraseña del usuario."""
    if not authenticate(username, old_password):
        return False

    users = load_users()
    hash_val, salt = _hash_password(new_password)
    users[username]["hash"] = hash_val
    users[username]["salt"] = salt
    save_users(users)
    return True
