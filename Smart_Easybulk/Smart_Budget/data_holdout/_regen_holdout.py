"""
Génère une base HOLDOUT pour validation externe du modèle.

DIFFÉRENCES avec data/_regen.py (training) :
  - seed = 999 (au lieu de 42)  → tirages complètement différents
  - mêmes groupes, mêmes quotas, mêmes recharges → pattern métier identique
  - mêmes règles saisonnières → le modèle DOIT toujours bien prédire

OBJECTIF : prouver que le modèle n'a pas mémorisé le training mais qu'il
a appris le PATTERN saisonnier + comportement métier.

Usage : python data_holdout/_regen_holdout.py
"""
import json, os, hashlib, random
from datetime import date, datetime, timedelta, timezone

# ⚠  Seed différent de celui du training (42) → tirages indépendants
random.seed(999)

DATA = os.path.dirname(os.path.abspath(__file__))

DATE_DEBUT = date(2023, 1, 1)
DATE_FIN   = date(2026, 4, 21)

GROUPES = [
    {"id":1,"name":"Prisons Nord","description":"safe_confortable","quota":10000,"quotaLoked":255,"quotaFree":9745,"status_id":1,"admin_id":2,"organization_id":1,"createdAt":"2023-01-01T08:00:00","updatedAt":"2026-04-20T18:00:00"},
    {"id":2,"name":"Prisons Centre","description":"danger_imminent","quota":1800,"quotaLoked":210,"quotaFree":1590,"status_id":1,"admin_id":3,"organization_id":1,"createdAt":"2023-01-01T08:00:00","updatedAt":"2026-04-20T18:00:00"},
    {"id":3,"name":"Prisons Sud","description":"critique_urgent","quota":600,"quotaLoked":80,"quotaFree":520,"status_id":1,"admin_id":4,"organization_id":1,"createdAt":"2023-01-01T08:00:00","updatedAt":"2026-04-20T18:00:00"},
    {"id":4,"name":"Prisons Est","description":"safe_actif","quota":6000,"quotaLoked":360,"quotaFree":5640,"status_id":1,"admin_id":5,"organization_id":1,"createdAt":"2023-01-01T08:00:00","updatedAt":"2026-04-20T18:00:00"},
    {"id":5,"name":"Test / Dev","description":"test_leger","quota":500,"quotaLoked":0,"quotaFree":500,"status_id":1,"admin_id":6,"organization_id":1,"createdAt":"2023-01-01T08:00:00","updatedAt":"2026-04-20T18:00:00"},
]

PRM_CAMPAIGN_TYPES = [
    {"code":"PROMO","value":"Promotion commerciale"},
    {"code":"NOTIF","value":"Notification transactionnelle"},
]

CTP = []
_ctp_id = 1
for g in GROUPES:
    for t in PRM_CAMPAIGN_TYPES:
        CTP.append({"id":_ctp_id,"enabled":True,"groupe_id":g["id"],"campaign_type":t["code"]})
        _ctp_id += 1
CTP_INDEX = {(c["groupe_id"], c["campaign_type"]): c["id"] for c in CTP}


CONSO_JOURNALIERE  = {1: 200,  2: 30,   3: 12,  4: 130,  5: 6}
RECHARGE_MENSUELLE = {1: 7150, 2: 1070, 3: 420, 4: 4650, 5: 210}

def multiplicateur_saison(d: date) -> float:
    m, j = d.month, d.day
    if m == 3 and j >= 10:          return 1.8
    if m == 4 and j <= 25:          return 2.2
    if m == 5 and j <= 3:           return 1.7
    if m == 6 and 14 <= j <= 20:    return 1.6
    if m == 9 and j <= 10:          return 1.4
    if m == 3 and 18 <= j <= 22:    return 1.3
    if m == 12 and j >= 24:         return 1.9
    if m == 1 and j <= 3:           return 1.5
    return 1.0

def iso_date(d: date) -> str:
    return d.isoformat()

budget_history = []
bh_id = 1

d = date(DATE_DEBUT.year, DATE_DEBUT.month, 5)
while d <= DATE_FIN:
    for g in GROUPES:
        amount = RECHARGE_MENSUELLE[g["id"]]
        if g["id"] == 2 and d.month in (3, 12):
            amount = int(amount * 1.5)
        budget_history.append({
            "id": bh_id,
            "groupe_id": g["id"],
            "modificationDate": iso_date(d),
            "amount": amount,
            "status_id": 1,
        })
        bh_id += 1
    y, m = d.year, d.month + 1
    if m > 12: y, m = y+1, 1
    d = date(y, m, 5)

# Consommations : mêmes règles que _regen.py mais seed différent
d = DATE_DEBUT
while d <= DATE_FIN:
    mult = multiplicateur_saison(d)
    is_we = d.weekday() >= 5
    for g in GROUPES:
        r = random.random()
        if r < 0.01:
            continue
        if r < 0.06:
            anomalie = random.uniform(0.20, 0.50)
        elif r < 0.09:
            anomalie = random.uniform(1.80, 2.50)
        else:
            anomalie = 1.0
        base = CONSO_JOURNALIERE[g["id"]]
        bruit = random.gauss(1.0, 0.15)
        bruit = max(0.5, min(1.5, bruit))
        conso = int(base * mult * bruit * anomalie)
        if g["id"] == 1 and is_we:
            conso = int(conso * 1.15)
        if conso <= 0:
            continue
        budget_history.append({
            "id": bh_id,
            "groupe_id": g["id"],
            "modificationDate": iso_date(d),
            "amount": -conso,
            "status_id": 2,
        })
        bh_id += 1
    d += timedelta(days=1)


def ts_tunis(d: date, h: int, mn: int = 0) -> str:
    return datetime(d.year, d.month, d.day, h, mn,
                    tzinfo=timezone(timedelta(hours=1))).isoformat()

def h_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:16]

CADENCE = {1: 14, 2: 30, 3: 45, 4: 18, 5: 60}

campagnes = []
c_id = 1
for g in GROUPES:
    gid = g["id"]
    cad = CADENCE[gid]
    d = DATE_DEBUT
    while d <= DATE_FIN:
        code = "PROMO" if (c_id % 2 == 0) else "NOTIF"
        ctp_id = CTP_INDEX[(gid, code)]
        mult = multiplicateur_saison(d)
        contacts = int(random.randint(80, 400) * mult)
        cost = contacts * random.randint(2, 4)
        used = int(cost * random.uniform(0.7, 0.95))
        lib  = f"{'Promo' if code=='PROMO' else 'Notif'} groupe {gid} {d.isoformat()}"
        campagnes.append({
            "id": c_id,
            "libelle": lib,
            "description": f"{'normal' if mult == 1.0 else 'saison'} - groupe {gid}",
            "dateDebut": ts_tunis(d, 9),
            "dateFin":   ts_tunis(d + timedelta(days=2), 3),
            "dureValidite": 48,
            "cost": cost,
            "budgetUsed": used,
            "nbrPage": random.randint(1, 4),
            "countContact": contacts,
            "deactivatedByGroup": False,
            "hash": h_hash(lib),
            "meOnly": False,
            "lastUpdateStatusAt": ts_tunis(d, 10),
            "createdAt": ts_tunis(d - timedelta(days=1), 8),
            "updatedAt": ts_tunis(d, 10),
            "status_id": 4,
            "campaign_type_permission_id": ctp_id,
        })
        c_id += 1
        d += timedelta(days=cad)


def save(name, obj):
    path = os.path.join(DATA, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  saved  {name}.json  ({len(obj)} records)")

save("groupes", GROUPES)
save("prm_campaign_type", PRM_CAMPAIGN_TYPES)
save("campaign_type_permission", CTP)
save("budget_history", budget_history)
save("campagnes", campagnes)

print()
print(f"HOLDOUT — Couverture: {DATE_DEBUT} -> {DATE_FIN}  (seed=999)")
print(f"  budget_history : {len(budget_history)} lignes")
print(f"  campagnes      : {len(campagnes)} lignes")
