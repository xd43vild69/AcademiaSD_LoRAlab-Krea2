"""Inicializar archivo de credenciales si no existe.

Se ejecuta en el entrypoint del Docker.
"""

import os
import sys
from auth_module import add_user, save_users, load_users


def init_default_credentials():
    """Crear usuario admin por defecto si no existen credenciales."""
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASS", "change_me_now")

    users = load_users()
    if users:
        print(f"[OK] Archivo de credenciales ya existe con {len(users)} usuario(s)")
        return True

    print(f"[*] Inicializando credenciales por defecto...")
    print(f"    Usuario: {admin_user}")
    print(f"    Contraseña: {admin_pass}")

    success = add_user(admin_user, admin_pass, role="admin")
    if success:
        print("[OK] Usuario administrador creado")
        return True
    else:
        print("[!] Error al crear usuario administrador", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = init_default_credentials()
    sys.exit(0 if success else 1)
