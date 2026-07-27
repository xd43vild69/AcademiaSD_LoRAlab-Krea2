"""Inicializar archivo de credenciales si no existe.

Se ejecuta en el entrypoint del Docker.
"""

import os
import sys
from auth_module import add_user, save_users, load_users


def init_default_credentials():
    """Crear o actualizar usuario admin por defecto."""
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASS", "admin")

    print(f"[*] Inicializando credenciales por defecto...")
    print(f"    Usuario: {admin_user}")

    success = add_user(admin_user, admin_pass, role="admin", overwrite=True)
    if success:
        print(f"[OK] Usuario administrador '{admin_user}' configurado correctamente.")
        return True
    else:
        print("[!] Error al crear usuario administrador", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = init_default_credentials()
    sys.exit(0 if success else 1)
