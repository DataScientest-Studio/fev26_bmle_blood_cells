"""Test infrastructure : Supabase (table TestCedric) + DagsHub (datalake images)."""

import json
import psycopg2
import requests
from collections import Counter

# ── Supabase ──────────────────────────────────────────────────────
print("=" * 50)
print("  Supabase — table TestCedric")
print("=" * 50)

conn = psycopg2.connect(
    host="aws-0-eu-west-1.pooler.supabase.com",
    port=5432,
    dbname="postgres",
    user="postgres.ckwlujqjmlvnnbsijxjw",
    password="Zewhog-goxnu1-ginbih",
    sslmode="require",
    connect_timeout=10,
)
cur = conn.cursor()
cur.execute('SELECT * FROM "TestCedric"')
rows = cur.fetchall()
cols = [desc[0] for desc in cur.description]

print(" | ".join(cols))
print("-" * 50)
for row in rows:
    print(" | ".join(str(v) for v in row))
print(f"\n{len(rows)} ligne(s)")
cur.close()
conn.close()

# ── DagsHub ───────────────────────────────────────────────────────
print()
print("=" * 50)
print("  DagsHub — datalake images (Source_100)")
print("=" * 50)

DAGSHUB_USER  = "Dumegan"
DAGSHUB_TOKEN = "618a61d0e0ea64cb347370c1625c08c37c119e6d"
DAGSHUB_REPO  = "Bloodcells-project"
BASE = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}"
AUTH = (DAGSHUB_USER, DAGSHUB_TOKEN)

# Vérifie que le token DagsHub fonctionne (accès au manifest DVC)
dvc_url = f"{BASE}/raw/main/Source_100.dvc"
r = requests.get(dvc_url, auth=AUTH, timeout=15)
r.raise_for_status()
print(f"Accès DagsHub OK (Source_100.dvc trouvé)")

# Compte les images depuis le manifest embarqué dans l'image
with open("source_100_manifest.json") as f:
    files = json.load(f)

classes = Counter(path.split("/")[0] for path in files)

print(f"Total images disponibles : {len(files)}")
print()
print(f"{'Classe':<20} {'Nb images':>10}")
print("-" * 32)
for cls, count in sorted(classes.items()):
    print(f"{cls:<20} {count:>10}")
