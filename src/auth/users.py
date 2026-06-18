"""Gestion des comptes utilisateurs (hash bcrypt, jamais de mot de passe en clair)."""

import bcrypt
import psycopg2

from .db import get_connection


def create_user(username: str, password: str) -> None:
    """Crée un utilisateur. Lève ValueError si le username existe déjà."""
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                    (username, password_hash),
                )
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise ValueError(f"L'utilisateur '{username}' existe déjà.")
        conn.commit()
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    """Vérifie le mot de passe d'un utilisateur. Retourne False si inconnu."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return False
    return bcrypt.checkpw(password.encode(), row[0].encode())


def list_usernames() -> list[str]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT username, created_at FROM users ORDER BY username")
            return cur.fetchall()
    finally:
        conn.close()


def delete_user(username: str) -> bool:
    """Supprime un utilisateur. Retourne True s'il existait."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()
