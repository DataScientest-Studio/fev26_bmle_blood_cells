"""Gestion des comptes utilisateurs Streamlit (table users sur Supabase).

Usage :
    python scripts/manage_users.py add <username>      # demande le mot de passe
    python scripts/manage_users.py list
    python scripts/manage_users.py delete <username>
"""

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from auth.users import create_user, delete_user, list_usernames  # noqa: E402


def cmd_add(username: str) -> None:
    password = getpass.getpass("Mot de passe : ")
    confirm = getpass.getpass("Confirmer    : ")
    if password != confirm:
        print("Les deux mots de passe ne correspondent pas.")
        sys.exit(1)
    if len(password) < 8:
        print("Le mot de passe doit faire au moins 8 caractères.")
        sys.exit(1)
    try:
        create_user(username, password)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    print(f"Utilisateur '{username}' créé.")


def cmd_list() -> None:
    rows = list_usernames()
    if not rows:
        print("Aucun utilisateur.")
        return
    for username, created_at in rows:
        print(f"  {username:<20} créé le {created_at:%Y-%m-%d}")


def cmd_delete(username: str) -> None:
    if delete_user(username):
        print(f"Utilisateur '{username}' supprimé.")
    else:
        print(f"Utilisateur '{username}' introuvable.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gestion des comptes utilisateurs Streamlit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Créer un utilisateur")
    p_add.add_argument("username")

    sub.add_parser("list", help="Lister les utilisateurs")

    p_del = sub.add_parser("delete", help="Supprimer un utilisateur")
    p_del.add_argument("username")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args.username)
    elif args.command == "list":
        cmd_list()
    elif args.command == "delete":
        cmd_delete(args.username)


if __name__ == "__main__":
    main()
