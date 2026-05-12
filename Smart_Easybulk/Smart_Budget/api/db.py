"""
db.py — Connexion MySQL réutilisable.

Pourquoi un context manager (`with get_cursor() as cur:`) ?
  - Ouvre une connexion à chaque requête (pas de pool pour la démo)
  - Commit / rollback automatique
  - Ferme proprement même si erreur
"""
from contextlib import contextmanager
import mysql.connector
from mysql.connector import Error

from config import DB_CONFIG


@contextmanager
def get_cursor(dictionary: bool = True):
    """
    Usage :
        with get_cursor() as cur:
            cur.execute("SELECT * FROM groupe")
            rows = cur.fetchall()
    """
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=dictionary)
        yield cur
        conn.commit()
    except Error as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()


def test_connection() -> bool:
    """Vérifie au démarrage que XAMPP/MySQL répond."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Error as e:
        print(f"  ⚠  Connexion MySQL échouée : {e}")
        return False
