"""
events_calendar.py
──────────────────────────────────────────────────────────────────
Construit le calendrier complet des événements qui influencent la
consommation de messages d'un groupe.

5 sources d'événements combinées :
  1. Calendrier islamique (Ramadan, Aïd) — dates variables via hijridate
  2. Événements fixes grégoriens (Réveillon, Rentrée, Fête nationale)
  3. Pics récurrents détectés automatiquement dans l'historique
  4. Anniversaires détectés depuis les libellés de campagnes
  5. Événements personnalisés (data/custom_events.json)

API publique (utilisée par feature_engineering) :
  - construire_calendrier(an_debut, an_fin) -> dict
  - get_evenement_pour_date(calendrier, date, groupe_id=None) -> dict
  - get_prochain_evenement(calendrier, date, horizon_jours=30) -> dict

ENTRÉE  : output/clean_budget_history.json
          output/clean_campagnes.json
          data/custom_events.json  (optionnel)
SORTIE  : output/events_calendar.json
          output/events_report.txt
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Chemins ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "output")
DATA = os.path.join(BASE, "data")

# ── Paramètres de détection ──────────────────────────────────────
# H1-FIX : seuil monté de 1.5 → 3.0. Avec 1.5× médiane et une saisonnalité
# forte (Ramadan ×1.8, Aïd ×2.2), 52/52 semaines étaient marquées comme pic
# → la feature saison_multiplicateur saturait à mean=3.07 au lieu de ~1.1.
# 3.0× ne déclenche que pour les vraies anomalies (Aïd, Réveillon...).
SEUIL_PIC_RECURRENT      = 3.0   # conso > 3.0× médiane → pic candidat
MIN_ANNEES_ANNIVERSAIRE  = 2     # apparition sur ≥ 2 années pour confirmer
FENETRE_AVANT_EVENEMENT  = 3     # jours avant : préparation
FENETRE_APRES_EVENEMENT  = 2     # jours après : suite

# ── Logger simple ────────────────────────────────────────────────
_log = []
def log(msg=""):   print(msg); _log.append(str(msg))
def title(t):      log(); log("┌" + "─"*64 + "┐"); log(f"│  {t:<62}│"); log("└" + "─"*64 + "┘")
def ok(m):         log(f"  ✓  {m}")
def info(m):       log(f"  ·  {m}")
def warn(m):       log(f"  ⚠  {m}")


# ═══════════════════════════════════════════════════════════════
# MODULE 1 — CALENDRIER ISLAMIQUE
# ═══════════════════════════════════════════════════════════════
# Ramadan, Aïd el-Fitr, Aïd el-Adha suivent le calendrier hégirien
# et se décalent d'environ 11 jours par an dans le calendrier grégorien.

def calculer_calendrier_islamique(annee_debut: int, annee_fin: int) -> list:
    """Calcule les dates exactes (via hijridate) ou approximatives (fallback)."""
    try:
        from hijridate import Hijri
        return _islamique_exact(annee_debut, annee_fin, Hijri)
    except ImportError:
        warn("hijridate non installé (pip install hijridate==2.3.0)")
        warn("→ utilisation du fallback approximatif (décalage 11 j/an)")
        return _islamique_approx(annee_debut, annee_fin)


def _islamique_exact(annee_debut: int, annee_fin: int, Hijri) -> list:
    """Dates exactes via hijridate. Hijri(year, month, day).to_gregorian()."""
    CONFIG = [
        # (nom, mois_hijri, jour_hijri, mult, description)
        ("Ramadan",     9,  1, 1.8, "Mois sacré — pic SMS religieux"),
        ("Aïd el-Fitr", 10, 1, 2.2, "Fête de la rupture du jeûne"),
        ("Aïd el-Adha", 12, 10, 2.0, "Fête du sacrifice"),
    ]
    resultats = []
    # On couvre large : 1444H ≈ 2022-2023, 1452H ≈ 2030-2031
    for annee_h in range(1444, 1452):
        for nom, mois_h, jour_h, mult, desc in CONFIG:
            try:
                debut = Hijri(annee_h, mois_h, jour_h).to_gregorian()
                # Ramadan : 29 ou 30 jours selon la lune ; autres fêtes : ~3 jours
                if nom == "Ramadan":
                    try:
                        fin = Hijri(annee_h, mois_h, 30).to_gregorian()
                    except Exception:
                        fin = Hijri(annee_h, mois_h, 29).to_gregorian()
                else:
                    fin = debut + timedelta(days=2)

                if annee_debut <= debut.year <= annee_fin:
                    resultats.append({
                        "nom": nom, "type": "islamique", "annee": debut.year,
                        "date_debut": str(debut), "date_fin": str(fin),
                        "duree_jours": (fin - debut).days + 1,
                        "multiplicateur": mult, "description": desc,
                    })
            except Exception as e:
                warn(f"hijridate H={annee_h} {nom} : {e}")
                continue
    return resultats


def _islamique_approx(annee_debut: int, annee_fin: int) -> list:
    """
    Fallback si hijridate absent. Utilise une formule générique :
    chaque événement islamique se décale d'environ -11 jours/an par
    rapport à une année de référence (2024). Précision : ±2 jours.
    """
    REFERENCE = {
        # nom : (date_debut_2024, duree_jours, multiplicateur, description)
        "Ramadan":     (date(2024, 3, 11), 30, 1.8, "Mois sacré (approx)"),
        "Aïd el-Fitr": (date(2024, 4, 10),  3, 2.2, "Rupture du jeûne (approx)"),
        "Aïd el-Adha": (date(2024, 6, 16),  3, 2.0, "Fête du sacrifice (approx)"),
    }
    resultats = []
    for annee in range(annee_debut, annee_fin + 1):
        decalage_jours = (annee - 2024) * -11   # -11 j par année en avance
        for nom, (ref, duree, mult, desc) in REFERENCE.items():
            debut = ref + timedelta(days=decalage_jours)
            fin   = debut + timedelta(days=duree - 1)
            if debut.year == annee:
                resultats.append({
                    "nom": nom, "type": "islamique", "annee": annee,
                    "date_debut": str(debut), "date_fin": str(fin),
                    "duree_jours": duree, "multiplicateur": mult,
                    "description": desc,
                })
    return resultats


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — ÉVÉNEMENTS FIXES GRÉGORIENS
# ═══════════════════════════════════════════════════════════════
def calendrier_evenements_fixes(annee_debut: int, annee_fin: int) -> list:
    """Événements à date fixe dans le calendrier grégorien."""
    MODELES = [
        # (nom, mois_debut, jour_debut, duree, mult, description, annee_fin_dans_suivante)
        ("Réveillon / Nouvel An",       12, 24, 10, 1.5, "Fêtes de fin d'année", True),
        ("Rentrée scolaire",             9,  1,  7, 1.3, "Rentrée — campagnes promo", False),
        ("Fête nationale tunisienne",    3, 20,  2, 1.4, "Fête de l'indépendance", False),
    ]
    resultats = []
    for annee in range(annee_debut, annee_fin + 1):
        for nom, m, j, duree, mult, desc, dans_suivante in MODELES:
            debut = date(annee, m, j)
            fin   = debut + timedelta(days=duree - 1)
            resultats.append({
                "nom": nom, "type": "fixe_gregorien", "annee": annee,
                "date_debut": str(debut), "date_fin": str(fin),
                "duree_jours": duree, "multiplicateur": mult,
                "description": desc,
            })
    return resultats


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — DÉTECTION DES PICS RÉCURRENTS
# ═══════════════════════════════════════════════════════════════
def detecter_pics_recurrents(df_budget: pd.DataFrame, groupe_id: int = None) -> list:
    """
    Détecte les semaines ISO qui dépassent systématiquement la médiane
    de consommation sur plusieurs années.
    """
    conso = df_budget[df_budget["status_id"] == 2].copy()
    if groupe_id is not None:
        conso = conso[conso["groupe_id"] == groupe_id]
    if conso.empty:
        return []

    conso["modificationDate"] = pd.to_datetime(conso["modificationDate"], errors="coerce")
    conso["montant_abs"]      = conso["amount"].abs()
    conso["semaine"]          = conso["modificationDate"].dt.isocalendar().week.astype(int)
    conso["annee"]            = conso["modificationDate"].dt.year

    par_semaine = (
        conso.groupby(["groupe_id", "semaine", "annee"])["montant_abs"]
             .sum().reset_index()
    )
    mediane = par_semaine["montant_abs"].median()
    if mediane == 0:
        return []

    resultats = []
    for sem in range(1, 54):
        sub = par_semaine[par_semaine["semaine"] == sem]
        if sub.empty:
            continue
        annees_pic = sub.loc[sub["montant_abs"] > mediane * SEUIL_PIC_RECURRENT, "annee"].tolist()
        if len(annees_pic) >= MIN_ANNEES_ANNIVERSAIRE:
            mult = round(
                sub.loc[sub["annee"].isin(annees_pic), "montant_abs"].mean() / mediane, 2
            )
            resultats.append({
                "semaine_iso": sem,
                "annees_detectees": sorted(annees_pic),
                "multiplicateur": mult,
                "fiabilite": len(annees_pic),
            })
    return resultats


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — DÉTECTION DES ANNIVERSAIRES DEPUIS LES CAMPAGNES
# ═══════════════════════════════════════════════════════════════
def detecter_anniversaires(df_campagnes: pd.DataFrame) -> list:
    """
    Détecte les (mois, jour) qui reviennent sur ≥2 années avec un volume
    de contacts supérieur à 1.2× la médiane.
    """
    if df_campagnes.empty:
        return []

    df = df_campagnes.copy()
    df["dateDebut"] = pd.to_datetime(df["dateDebut"], utc=True, errors="coerce")
    df = df.dropna(subset=["dateDebut"])
    df["mois"]  = df["dateDebut"].dt.month
    df["jour"]  = df["dateDebut"].dt.day
    df["annee"] = df["dateDebut"].dt.year

    mediane_contacts = df["countContact"].median()
    if mediane_contacts == 0:
        return []

    resultats = []
    for gid in df["groupe_id"].unique():
        sous = df[df["groupe_id"] == gid]
        par_date = (
            sous.groupby(["mois", "jour", "annee"])
                .agg(
                    nb_camps   =("id",           "count"),
                    tot_contact=("countContact", "sum"),
                    libelles   =("libelle",      lambda x: " | ".join(x.dropna().unique()[:3]))
                )
                .reset_index()
        )
        for (mois, jour), grp in par_date.groupby(["mois", "jour"]):
            if len(grp) < MIN_ANNEES_ANNIVERSAIRE:
                continue
            if grp["tot_contact"].mean() < mediane_contacts * 1.2:
                continue
            texte = " ".join(grp["libelles"].tolist()).lower()
            resultats.append({
                "groupe_id"      : int(gid),
                "mois"           : int(mois),
                "jour"           : int(jour),
                "label_detecte"  : f"{jour:02d}/{mois:02d}",
                "nom_evenement"  : _extraire_nom_evenement(texte, mois, jour),
                "annees"         : sorted(grp["annee"].tolist()),
                "contacts_moyen" : round(grp["tot_contact"].mean()),
                "multiplicateur" : round(grp["tot_contact"].mean() / mediane_contacts, 2),
                "description"    : "Anniversaire/événement récurrent détecté",
            })
    return resultats


def _extraire_nom_evenement(texte: str, mois: int, jour: int) -> str:
    """Identifie un nom d'événement depuis les libellés (FR/AR/EN)."""
    MOTS_CLES = {
        "Anniversaire" : ["anniversaire", "anniv", "birthday", "ذكرى", "عيد ميلاد"],
        "Ramadan"      : ["ramadan", "رمضان"],
        "Eid"          : ["eid", "عيد", "aid", "aïd"],
        "Solde"        : ["solde", "promo", "promotion", "offre", "réduction"],
        "Noel"         : ["noël", "noel", "christmas", "réveillon", "bonne année"],
        "Rentrée"      : ["rentrée", "rentree", "back to school"],
        "Fête nationale": ["fête nationale", "national", "indépendance", "استقلال"],
        "Fin d'année"  : ["fin d'année", "year end", "bilan"],
    }
    for nom, mots in MOTS_CLES.items():
        if any(mot in texte for mot in mots):
            return nom
    MOIS_NOMS = ["", "Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                 "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    return f"Événement récurrent {jour} {MOIS_NOMS[mois]}"


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — ÉVÉNEMENTS PERSONNALISÉS
# ═══════════════════════════════════════════════════════════════
def charger_evenements_custom() -> list:
    """Charge data/custom_events.json s'il existe."""
    chemin = os.path.join(DATA, "custom_events.json")
    if not os.path.exists(chemin):
        return []
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    info(f"custom_events.json : {len(data)} événement(s)")
    return data


# ═══════════════════════════════════════════════════════════════
# CONSTRUCTION DU CALENDRIER COMPLET
# ═══════════════════════════════════════════════════════════════
def construire_calendrier(annee_debut: int, annee_fin: int) -> dict:
    """Assemble le calendrier depuis les 5 sources. Sortie = dict JSON-compatible."""
    path_b = os.path.join(OUT, "clean_budget_history.json")
    path_c = os.path.join(OUT, "clean_campagnes.json")
    df_b = pd.read_json(path_b, orient="records") if os.path.exists(path_b) else pd.DataFrame()
    df_c = pd.read_json(path_c, orient="records") if os.path.exists(path_c) else pd.DataFrame()

    title("MODULE 1 — CALENDRIER ISLAMIQUE")
    islamiques = calculer_calendrier_islamique(annee_debut, annee_fin)
    for e in islamiques:
        ok(f"{e['annee']}  {e['nom']:<14}  {e['date_debut']} → {e['date_fin']}  "
           f"(×{e['multiplicateur']})")

    title("MODULE 2 — ÉVÉNEMENTS FIXES GRÉGORIENS")
    fixes = calendrier_evenements_fixes(annee_debut, annee_fin)
    for nom in sorted({e["nom"] for e in fixes}):
        ok(nom)

    title("MODULE 3 — DÉTECTION PICS RÉCURRENTS")
    pics = detecter_pics_recurrents(df_b) if not df_b.empty else []
    if pics:
        for p in pics:
            ok(f"Semaine ISO {p['semaine_iso']:>2}  ×{p['multiplicateur']}  "
               f"({p['fiabilite']} ans : {p['annees_detectees']})")
    else:
        info("Aucun pic récurrent (historique insuffisant ou absent)")

    title("MODULE 4 — ANNIVERSAIRES DEPUIS CAMPAGNES")
    anniversaires = detecter_anniversaires(df_c) if not df_c.empty else []
    if anniversaires:
        for a in anniversaires:
            ok(f"Groupe {a['groupe_id']}  {a['nom_evenement']:<18}  "
               f"{a['label_detecte']}  ×{a['multiplicateur']}")
    else:
        info("Aucun anniversaire détecté")

    title("MODULE 5 — ÉVÉNEMENTS PERSONNALISÉS")
    custom = charger_evenements_custom()
    if not custom:
        info("data/custom_events.json absent — aucun événement custom")

    return {
        "annee_debut": annee_debut,
        "annee_fin"  : annee_fin,
        "islamiques" : islamiques,
        "fixes_gregoriens": fixes,
        "pics_recurrents" : pics,
        "anniversaires"   : anniversaires,
        "custom"          : custom,
        "meta": {
            "nb_islamiques"     : len(islamiques),
            "nb_fixes"          : len(fixes),
            "nb_pics_recurrents": len(pics),
            "nb_anniversaires"  : len(anniversaires),
            "nb_custom"         : len(custom),
        },
    }


# ═══════════════════════════════════════════════════════════════
# API — Fonction utilisée par feature_engineering.py
# ═══════════════════════════════════════════════════════════════
# Cache : la liste unifiée est coûteuse à construire, on la réutilise.
_cache_unified = {"id_cal": None, "liste": None}


def _construire_liste_unifiee(calendrier: dict) -> list:
    """
    Transforme les 5 sources d'événements en UNE seule liste unifiée
    avec un schéma commun. Simplifie get_evenement_pour_date.

    Schéma unifié :
      {nom, type, debut (date), fin (date), multiplicateur, groupe_id (opt)}
    """
    liste = []

    for e in calendrier.get("islamiques", []):
        liste.append({
            "nom": e["nom"], "type": "islamique",
            "debut": pd.Timestamp(e["date_debut"]).date(),
            "fin"  : pd.Timestamp(e["date_fin"]).date(),
            "multiplicateur": e["multiplicateur"], "groupe_id": None,
        })

    for e in calendrier.get("fixes_gregoriens", []):
        liste.append({
            "nom": e["nom"], "type": "fixe",
            "debut": pd.Timestamp(e["date_debut"]).date(),
            "fin"  : pd.Timestamp(e["date_fin"]).date(),
            "multiplicateur": e["multiplicateur"], "groupe_id": None,
        })

    # Pics : définis par semaine ISO → étendus sur toutes les années
    an_debut = calendrier.get("annee_debut", 2023)
    an_fin   = calendrier.get("annee_fin",   2027)
    for p in calendrier.get("pics_recurrents", []):
        for annee in range(an_debut, an_fin + 1):
            try:
                lundi = date.fromisocalendar(annee, p["semaine_iso"], 1)
            except ValueError:
                continue   # semaine 53 n'existe pas toutes les années
            liste.append({
                "nom": f"Pic récurrent S{p['semaine_iso']}",
                "type": "pic_detecte",
                "debut": lundi, "fin": lundi + timedelta(days=6),
                "multiplicateur": p["multiplicateur"], "groupe_id": None,
            })

    # Anniversaires : (mois, jour) annuels → étendus
    for a in calendrier.get("anniversaires", []):
        for annee in range(an_debut, an_fin + 1):
            try:
                d = date(annee, a["mois"], a["jour"])
            except ValueError:
                continue
            liste.append({
                "nom": a["nom_evenement"], "type": "anniversaire",
                "debut": d, "fin": d + timedelta(days=1),
                "multiplicateur": a["multiplicateur"],
                "groupe_id": a["groupe_id"],
            })

    # Custom annuels
    for e in calendrier.get("custom", []):
        if e.get("recurrence") != "annuel":
            continue
        for annee in range(an_debut, an_fin + 1):
            try:
                debut = date(annee, e["mois"], e["jour"])
                fin   = debut + timedelta(days=e.get("duree_jours", 1) - 1)
            except (ValueError, KeyError):
                continue
            liste.append({
                "nom": e["nom"], "type": "custom",
                "debut": debut, "fin": fin,
                "multiplicateur": e.get("multiplicateur", 1.3),
                "groupe_id": None,
            })
    return liste


def get_evenement_pour_date(calendrier: dict, d: date, groupe_id: int = None) -> dict:
    """
    Retourne l'événement actif le plus important (plus fort multiplicateur)
    pour la date d, ou 'normal' si aucun événement.

    Réponse :
      {nom, type, multiplicateur, jours_avant, actif}
    """
    # Cache : ne reconstruit la liste que si le calendrier change
    id_cal = id(calendrier)
    if _cache_unified["id_cal"] != id_cal:
        _cache_unified["id_cal"] = id_cal
        _cache_unified["liste"]  = _construire_liste_unifiee(calendrier)

    defaut = {"nom": "normal", "type": "normal", "multiplicateur": 1.0,
              "jours_avant": None, "actif": False}

    meilleur = defaut
    for e in _cache_unified["liste"]:
        # Filtre par groupe si applicable (anniversaires)
        if e["groupe_id"] is not None and groupe_id is not None:
            if e["groupe_id"] != groupe_id:
                continue

        debut_elargi = e["debut"] - timedelta(days=FENETRE_AVANT_EVENEMENT)
        fin_elargi   = e["fin"]   + timedelta(days=FENETRE_APRES_EVENEMENT)

        if debut_elargi <= d <= fin_elargi:
            if e["multiplicateur"] > meilleur["multiplicateur"]:
                meilleur = {
                    "nom"           : e["nom"],
                    "type"          : e["type"],
                    "multiplicateur": e["multiplicateur"],
                    "jours_avant"   : max(0, (e["debut"] - d).days) if d < e["debut"] else 0,
                    "actif"         : e["debut"] <= d <= e["fin"],
                }
    return meilleur


def get_prochain_evenement(calendrier: dict, d: date,
                           horizon_jours: int = 30,
                           groupe_id: int = None) -> dict:
    """
    Retourne le prochain événement dans les N jours, ou {nom: Aucun}.

    BUG-FIX (B1) : si groupe_id est fourni, on prend en compte les
    anniversaires/événements spécifiques à ce groupe (détectés depuis
    les libellés de campagnes). Sans ça, tous les groupes voyaient le
    même prochain événement (uniquement les événements globaux :
    Ramadan, Aïd, Réveillon, Fête nationale).
    """
    for delta in range(1, horizon_jours + 1):
        evt = get_evenement_pour_date(
            calendrier, d + timedelta(days=delta), groupe_id=groupe_id
        )
        if evt["nom"] != "normal":
            return {**evt, "dans_jours": delta,
                    "date_estimee": str(d + timedelta(days=delta))}
    return {"nom": "Aucun", "dans_jours": None, "date_estimee": None}


# ═══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Calendrier glissant : toujours ancré sur l'année courante pour éviter
    # l'obsolescence. −3 ans : profondeur pour detecter_pics_recurrents et
    # l'approche C (comparaison années précédentes). +2 ans : marge future
    # pour l'horizon 30 jours de get_prochain_evenement.
    ANNEE_COURANTE = date.today().year
    ANNEE_DEBUT    = ANNEE_COURANTE - 3
    ANNEE_FIN      = ANNEE_COURANTE + 2

    title("CONSTRUCTION DU CALENDRIER DES ÉVÉNEMENTS")
    info(f"Année courante : {ANNEE_COURANTE}")
    info(f"Plage couverte : {ANNEE_DEBUT} → {ANNEE_FIN} (glissante)")

    calendrier = construire_calendrier(ANNEE_DEBUT, ANNEE_FIN)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "events_calendar.json"), "w", encoding="utf-8") as f:
        json.dump(calendrier, f, indent=2, ensure_ascii=False, default=str)
    ok("events_calendar.json  →  calendrier sauvegardé")

    title("RÉSUMÉ")
    m = calendrier["meta"]
    log(f"  Islamiques    : {m['nb_islamiques']:>3}")
    log(f"  Fixes         : {m['nb_fixes']:>3}")
    log(f"  Pics détectés : {m['nb_pics_recurrents']:>3}")
    log(f"  Anniversaires : {m['nb_anniversaires']:>3}")
    log(f"  Custom        : {m['nb_custom']:>3}")

    title("DÉMONSTRATION — get_evenement_pour_date()")
    dates_test = [
        date(2024, 3, 13),   # Ramadan 2024
        date(2025, 3, 5),    # Ramadan 2025
        date(2024, 4, 10),   # Aïd el-Fitr 2024
        date(2024, 12, 25),  # Réveillon
        date(2024, 6, 15),   # Normal
        date(2024, 9, 3),    # Rentrée
    ]
    for d in dates_test:
        evt = get_evenement_pour_date(calendrier, d)
        marqueur = "[ACTIF]" if evt["actif"] else ""
        log(f"  {d}  →  {evt['nom']:<22}  ×{evt['multiplicateur']:.1f}  {marqueur}")

    with open(os.path.join(OUT, "events_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log))
    ok("events_report.txt  →  rapport sauvegardé")

    log()
    log("══════════════════════════════════════════════════════════════════")
    log("  CALENDRIER TERMINÉ  —  lancer ensuite : python 02_feature_engineering.py")
    log("══════════════════════════════════════════════════════════════════")
