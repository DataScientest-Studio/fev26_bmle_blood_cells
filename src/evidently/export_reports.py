"""
Exporte les rapports de drift HTML depuis Supabase vers des fichiers locaux.

Usage:
    python -m src.evidently.export_reports              # Exporte le dernier rapport
    python -m src.evidently.export_reports --all        # Exporte tous les rapports
    python -m src.evidently.export_reports --limit 5    # Exporte les 5 derniers
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# Vérifier les variables d'environnement
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_PORT = int(os.getenv("SUPABASE_PORT", 6543))
SUPABASE_DB = os.getenv("SUPABASE_DB")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

if not all([SUPABASE_HOST, SUPABASE_DB, SUPABASE_USER, SUPABASE_PASSWORD]):
    print("❌ Variables d'environnement Supabase manquantes dans .env")
    sys.exit(1)

# Dossier de destination
OUTPUT_DIR = Path(__file__).resolve().parent / "reports_html"
OUTPUT_DIR.mkdir(exist_ok=True)


def export_reports(limit: int = 1, all_reports: bool = False):
    """
    Exporte les rapports HTML depuis Supabase.

    Parameters
    ----------
    limit : int
        Nombre de rapports à exporter (défaut: 1 = dernier)
    all_reports : bool
        Si True, exporte tous les rapports (ignore limit)
    """
    try:
        conn = psycopg2.connect(
            host=SUPABASE_HOST,
            port=SUPABASE_PORT,
            dbname=SUPABASE_DB,
            user=SUPABASE_USER,
            password=SUPABASE_PASSWORD,
            sslmode="require",
        )
        cur = conn.cursor()

        # Requête SQL
        if all_reports:
            query = """
                SELECT id, created_at, model_version, data_drift_score,
                       pred_drift_score, report_html
                FROM drift_reports
                ORDER BY created_at DESC
            """
            cur.execute(query)
        else:
            query = """
                SELECT id, created_at, model_version, data_drift_score,
                       pred_drift_score, report_html
                FROM drift_reports
                ORDER BY created_at DESC
                LIMIT %s
            """
            cur.execute(query, (limit,))

        reports = cur.fetchall()
        cur.close()
        conn.close()

        if not reports:
            print("❌ Aucun rapport trouvé dans Supabase.")
            return

        print(f"\n✓ {len(reports)} rapport(s) trouvé(s)\n")

        for report_id, created_at, model_version, data_drift, pred_drift, html in reports:
            # Créer un nom de fichier avec timestamp
            timestamp = created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"drift_report_{report_id}_{timestamp}.html"
            filepath = OUTPUT_DIR / filename

            # Sauvegarder le HTML
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)

            # Afficher les infos
            print(f"📄 {filename}")
            print(f"   ID: {report_id}")
            print(f"   Date: {created_at}")
            print(f"   Modèle: {model_version or 'non spécifié'}")
            print(f"   Data drift: {data_drift:.4f}" if data_drift else "   Data drift: N/A")
            print(f"   Pred drift: {pred_drift:.4f}" if pred_drift else "   Pred drift: N/A")
            print(f"   Chemin: {filepath}\n")

        print(f"✓ Rapports exportés dans: {OUTPUT_DIR}")
        print(f"✓ Ouvrir dans le navigateur: file://{OUTPUT_DIR.absolute()}")

    except psycopg2.Error as e:
        print(f"❌ Erreur Supabase: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Exporte les rapports de drift HTML depuis Supabase"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exporte tous les rapports (ignore --limit)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Nombre de rapports à exporter (défaut: 1)",
    )

    args = parser.parse_args()

    export_reports(limit=args.limit, all_reports=args.all)
