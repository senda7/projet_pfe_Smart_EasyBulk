# ==========================Importation des bibliothèques standards==========================================
import json
import warnings
import os
import logging
import time
from threading import Thread
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib  # pour remplacer pickle

warnings.filterwarnings('ignore')
# ================================CONFIGURATION GÉNÉRALE =============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8'),
    ], )
log = logging.getLogger('training')
_BASE_DIR = Path(__file__).resolve().parent
CONFIG = {
    'USE_MONGODB': True,
    'MONGO_URI': 'mongodb://localhost:27017',
    'MONGO_DB': 'test',
    'MONGO_COL': 'contel',
    'OUTPUT_DIR': str(_BASE_DIR / 'outputs'),
    'MODEL_PKL': str(_BASE_DIR / 'outputs' / 'model_assets.joblib'),
    'MODEL_XGB': str(_BASE_DIR / 'outputs' / 'xgboost_model.json'),
    'PREDICTIONS_CSV': str(_BASE_DIR / 'outputs' / 'contact_predictions.csv'),
    'METRICS_JSON': str(_BASE_DIR / 'outputs' / 'eval_metrics.json'),
    'SCHEDULE_TIME': '02:00',
    'RETRAIN_INTERVAL_DAYS': 14,
    'RUN_NOW': True,
    'ENABLE_HYPERPARAMETER_TUNING': True,
    'TUNING_CV_FOLDS': 3,        # nb de folds CV dans chaque trial Optuna
    'TUNING_N_TRIALS': 50,       # nb de trials Optuna
}
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

VALID_STATUSES = {1, 2, 4, 8}        # statuts DLR valides à garder
STATUSES_TO_CLEAN = {0, 16, 34}      # statuts supprimés lors du nettoyage des données
DLR_NORM = {1: 1, 2: 2, 4: 2, 8: 2}  # normalisation des statuts
DLR_SUCCESS = frozenset({1})         # les SMS livrés
DLR_FAILURE = frozenset({2})         # les SMS non livrés
DLR_NA = frozenset({0, 16, 34})
DLR_TRANSIT = frozenset({4, 8})
DLR_FINAL = DLR_SUCCESS | DLR_FAILURE   # les statuts finaux après normalisation

XGB_FEATURE_COLS = [
    'n_succes_hist',               # nb succès total
    'jours_depuis_succes',         # ancienneté du dernier succès
    'score_recence_pondere',       # time-decay succès/échecs
    'n_succes_30j',                # succès récents
    'taux_succes_30j',             # taux récent
    'echecs_consecutifs_fin',      # échecs consécutifs récents
    'n_envois_hist',               # volume total d'envois
    'jours_depuis_dernier_envoi',  # inactivité récente
]

RSF_FEATURE_COLS = [
    'jours_depuis_succes',         # durée depuis dernier succès → corrélé à duree_survie
    'score_recence_pondere',       # signal composite joignabilité
    'n_succes_hist',               # quantité de succès passés
    'taux_succes_30j',             # taux récent
    'echecs_consecutifs_fin',      # dégradation finale
    'jours_depuis_dernier_envoi',  # inactivité
]
RSF_EXTRA_COLS = [
    'evenement_na',
    'duree_survie_jours',
]
SCORE_THRESHOLDS = {
    'available': 50,   # seuil Available
    'suspected': 38,   # seuil Suspected
}

# =========================Extraction des données ================================
def load_from_mongodb():
    from pymongo import MongoClient
    client = MongoClient(CONFIG['MONGO_URI'])
    col = client[CONFIG['MONGO_DB']][CONFIG['MONGO_COL']]
    cursor = col.find({}, {'_id': 0}).batch_size(10000)
    for doc in cursor:
        yield doc
    client.close()
# ============================== Filtrage et validation =============================
def filter_and_validate(generator):

    DATE_FMT = '%Y-%m-%d %H:%M:%S'
    valid = []
    stats = {
        'total_raw': 0, 'invalid_msisdn': 0,
        'invalid_date': 0, 'unknown_status': 0,
        'cleaned_status': 0,
        'duplicate_sms': 0, 'valid': 0,
    }
    seen_sms = set()

    for r in generator:
        stats['total_raw'] += 1

        # ---------Validation MSISDN-------
        msisdn = str(r.get('msisdn', '')).strip()
        if not msisdn or len(msisdn) < 8 or not msisdn.lstrip('+').isdigit():
            stats['invalid_msisdn'] += 1
            continue

        status = r.get('status')
        date_str = r.get('status_last_updated_at', '')

        if status is None or not date_str:
            stats['unknown_status'] += 1
            continue

        if status not in VALID_STATUSES:
            if status in STATUSES_TO_CLEAN:
                stats['cleaned_status'] += 1
            else:
                stats['unknown_status'] += 1
            continue

        try:
            dt = datetime.strptime(date_str, DATE_FMT)
        except (ValueError, TypeError):
            stats['invalid_date'] += 1
            continue


        key = (msisdn, status, dt)
        if key in seen_sms:
            stats['duplicate_sms'] += 1
            continue

        seen_sms.add(key)
        valid.append({'msisdn': msisdn, 'status': status, 'dt': dt})
        stats['valid'] += 1

    log.info(f" {stats['valid']} valides | {stats['total_raw'] - stats['valid']} rejetés")
    log.info(f"  MSISDN invalide:{stats['invalid_msisdn']} | Date invalide:{stats['invalid_date']}")
    log.info(f"  Statuts nettoyés (0,16,34):{stats['cleaned_status']} | Statuts inconnus:{stats['unknown_status']} | SMS dupliqués:{stats['duplicate_sms']}")
    return valid, stats


# ==========================Normalisation des statuts DLR==================================================
def normalise_statuses(valid_records):
    normalised_count = 0
    for rec in valid_records:
        orig = rec['status']
        norm = DLR_NORM.get(orig, orig)
        if norm != orig:
            normalised_count += 1
        rec['status'] = norm

    log.info(f" {normalised_count} statuts normalisés (4→2, 8→2)")

    groups = defaultdict(list)
    for r in valid_records:
        groups[r['msisdn']].append(r)

    contacts = []
    for msisdn, recs in groups.items():
        recs.sort(key=lambda x: x['dt'])
        history = [{'status': r['status'], 'dt': r['dt']} for r in recs]
        contacts.append({
            'msisdn': msisdn,
            'history': history,
            'final_status': recs[-1]['status'],
            'n_sends': len(recs),
        })

    log.info(f"BLOC 1C — {len(contacts)} MSISDNs uniques")
    return contacts


def extract_and_clean(raw_generator):
    valid, _ = filter_and_validate(raw_generator)
    return normalise_statuses(valid)


# =========================== Feature Engineering  ========================================================
def _label_from_last_envoi(last_status, hist_past, last_dt, now):

    n_succes_past = sum(1 for h in hist_past if h['status'] in DLR_SUCCESS)

    if last_status in DLR_SUCCESS:
        return 'Available'

    if n_succes_past > 0:
        return 'Suspected'

    jours_depuis_last = (now - last_dt).days
    if not hist_past and jours_depuis_last < 30:
        return 'Suspected'

    if hist_past:
        last_past_dt = hist_past[-1]['dt']
        jours_depuis_dernier_past = (now - last_past_dt).days
        if jours_depuis_dernier_past < 14:
            return 'Suspected'
    return 'NA'


def build_features(contacts):
    now = datetime.now()
    rows = []
    for c in contacts:
        hist = c['history']
        if not hist:
            continue

        # ── Séparation temporelle  ─────────────────────────────────────
        last_entry = hist[-1]
        hist_past = hist[:-1]
        last_status = last_entry['status']
        last_dt = last_entry['dt']

        # Label = f(statut du dernier envoi, historique passé, ancienneté)
        label = _label_from_last_envoi(last_status, hist_past, last_dt, now)

        # ── Features calculées sur hist_past uniquement ───────────────────────
        n_hist = len(hist_past)
        statuses_past = [h['status'] for h in hist_past]
        dts_past = [h['dt'] for h in hist_past]

        # Counts absolus (pas de ratios → pas de leakage)
        n_envois_hist = n_hist
        n_succes_hist = sum(1 for s in statuses_past if s in DLR_SUCCESS)
        n_echecs_hist = sum(1 for s in statuses_past if s in DLR_FAILURE)

        # Séquence max d'échecs consécutifs sur tout l'historique passé
        echecs_consecutifs_max = 0
        cur = 0
        for s in statuses_past:
            cur = cur + 1 if s in DLR_FAILURE else 0
            echecs_consecutifs_max = max(echecs_consecutifs_max, cur)

        # Séquence d'échecs consécutifs EN FIN d'historique (signal de dégradation récente)
        echecs_consecutifs_fin = 0
        for s in reversed(statuses_past):
            if s in DLR_FAILURE:
                echecs_consecutifs_fin += 1
            else:
                break

        # Ancienneté du dernier envoi connu (= last_dt, date réelle)
        jours_depuis_dernier_envoi = max(0, (now - last_dt).days)

        # Ancienneté du dernier succès dans l'historique PASSÉ
        last_ok_dt = next(
            (h['dt'] for h in reversed(hist_past) if h['status'] in DLR_SUCCESS), None
        )
        jours_depuis_succes = int((now - last_ok_dt).days) if last_ok_dt else 999

        # Score de récence pondéré (time-decay λ=0.02) sur l'historique passé
        # Note: calculé sur hist_past → pas de leakage possible
        DECAY_LAMBDA = 0.02  # demi-vie ≈ 35 jours
        score_recence_pondere = 0.0
        for h in hist_past:
            age = max(0, (now - h['dt']).days)
            w = np.exp(-DECAY_LAMBDA * age)
            if h['status'] in DLR_SUCCESS:
                score_recence_pondere += w
            elif h['status'] in DLR_FAILURE:
                penalty = 1.5 if age < 7 else 0.8
                score_recence_pondere -= w * penalty
        score_recence_pondere /= max(1.0, np.sqrt(max(1, n_envois_hist)))
        score_recence_pondere = float(np.clip(score_recence_pondere, -10.0, 10.0))

        # Cadence inter-envoi (fréquence d'envoi passée)
        if len(dts_past) > 1:
            span = max(1, (dts_past[-1] - dts_past[0]).days)
            freq_inter_envoi_jours = round(span / (len(dts_past) - 1), 2)
        else:
            freq_inter_envoi_jours = 0.0

        # Durée d'observation totale de l'historique passé
        if len(dts_past) > 1:
            duree_observation_jours = max(1, (dts_past[-1] - dts_past[0]).days)
        elif len(dts_past) == 1:
            duree_observation_jours = max(1, (now - dts_past[0]).days)
        else:
            duree_observation_jours = 1

        # ── Features RSF — événement et durée de survie ───────────────────────
        # evenement_na = True si dégradation : dernier envoi échoué OU ≥2 échecs sur 3 derniers
        n_echecs_recent = sum(1 for h in hist[-3:] if h['status'] in DLR_FAILURE)
        evenement_na = (last_status in DLR_FAILURE) or (n_echecs_recent >= 2)


        last_ok_dt_full = next(
            (h['dt'] for h in reversed(hist) if h['status'] in DLR_SUCCESS), None
        )
        if last_ok_dt_full is not None:
            # Durée depuis le dernier succès (tout historique) jusqu'à maintenant
            duree_survie_jours = max(1, int((now - last_ok_dt_full).days))
        else:
            # Jamais de succès : durée depuis le 1er envoi connu jusqu'à maintenant
            duree_survie_jours = max(1, int((now - hist[0]['dt']).days))

        # ── Fenêtre 30j ──────────────────────────────────────────────────────
        cutoff_30j = last_dt - timedelta(days=30)
        hist_30j = [h for h in hist_past if h['dt'] >= cutoff_30j]
        n_succes_30j = sum(1 for h in hist_30j if h['status'] in DLR_SUCCESS)
        n_echecs_30j = sum(1 for h in hist_30j if h['status'] in DLR_FAILURE)
        n_total_30j = len(hist_30j)
        taux_succes_30j = round((n_succes_30j + 1) / (n_total_30j + 2), 4)

        rows.append({
            'msisdn': c['msisdn'],
            # features XGBoost essentielles
            'n_envois_hist': n_envois_hist,
            'n_succes_hist': n_succes_hist,
            'echecs_consecutifs_fin': echecs_consecutifs_fin,
            'jours_depuis_dernier_envoi': jours_depuis_dernier_envoi,
            'jours_depuis_succes': jours_depuis_succes,
            'score_recence_pondere': round(score_recence_pondere, 4),
            'n_succes_30j': n_succes_30j,
            'taux_succes_30j': taux_succes_30j,
            # label et RSF
            'label': label,
            'evenement_na': evenement_na,
            'duree_survie_jours': duree_survie_jours,
        })

    df = pd.DataFrame(rows)
    log.info(f"Features → {len(df)} contacts | labels : {df['label'].value_counts().to_dict()}")
    counts = df['label'].value_counts()
    if counts.min() < 50:
        log.warning(
            f"Classe minoritaire très faible ({counts.min()} exemples) → "
            f"résultats potentiellement peu fiables pour cette classe"
        )
    return df

# ======================== Encodage des labels ====================================================
def encode_labels(df_real):
    df = df_real.copy()
    df['source'] = 'real'
    ordered = ['Available', 'Suspected', 'NA']
    present = [l for l in ordered if l in df['label'].unique()]
    label_map = {lbl: i for i, lbl in enumerate(present)}
    df['label_enc'] = df['label'].map(label_map)
    num_class = len(label_map)
    df.attrs['label_map'] = label_map
    log.info(f"Dataset → {len(df)} contacts | labels={df['label'].value_counts().to_dict()}")
    log.info(f"label_map dynamique : {label_map} | num_class={num_class}")
    return df, label_map

# ======================= XGBoost Hyperparameter ====================================================
def _default_xgb_params():
    return dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.15,
        reg_alpha=0.15,
        reg_lambda=1.2,
    )

def _tune_xgboost_optuna(X_tr, y_tr, X_val, y_val, num_class, sw_tr=None, sw_val=None, n_trials=50):
    import optuna
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_cv = np.vstack([X_tr, X_val])
    y_cv = np.concatenate([y_tr, y_val])

    from sklearn.utils.class_weight import compute_sample_weight

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 200, 600, step=50),
            max_depth=trial.suggest_int('max_depth', 3, 6),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.10, log=True),
            subsample=trial.suggest_float('subsample', 0.60, 0.95),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.50, 0.95),
            min_child_weight=trial.suggest_int('min_child_weight', 3, 15),
            gamma=trial.suggest_float('gamma', 0.0, 0.5),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.5, 3.0),
        )
        skf = StratifiedKFold(n_splits=CONFIG.get('TUNING_CV_FOLDS', 3), shuffle=True, random_state=42)
        fold_scores = []
        for fold_tr_idx, fold_val_idx in skf.split(X_cv, y_cv):
            Xf_tr, Xf_val = X_cv[fold_tr_idx], X_cv[fold_val_idx]
            yf_tr, yf_val = y_cv[fold_tr_idx], y_cv[fold_val_idx]
            swf_tr = compute_sample_weight('balanced', yf_tr)
            swf_val = compute_sample_weight('balanced', yf_val)
            m = xgb.XGBClassifier(
                **params,
                objective='multi:softprob', num_class=num_class,
                eval_metric='mlogloss', early_stopping_rounds=20,
                random_state=42, verbosity=0,
            )
            m.fit(
                Xf_tr, yf_tr,
                sample_weight=swf_tr,
                eval_set=[(Xf_val, yf_val)],
                sample_weight_eval_set=[swf_val.tolist()],
                verbose=False,
            )
            fold_scores.append(f1_score(yf_val, m.predict(Xf_val), average='macro', zero_division=0))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    log.info(f"Optuna tuning terminé — {n_trials} trials | meilleur F1-macro CV-3fold={study.best_value:.4f}")
    log.info(f"  Meilleurs hyperparamètres : {best}")
    return best


def _tune_xgboost_grid(X_tr, y_tr, X_val, y_val, num_class, sw_tr=None, sw_val=None):
    from sklearn.metrics import f1_score
    from itertools import product

    grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.05, 0.08],
        'min_child_weight': [5, 8, 12],
    }

    best_f1, best_params = -1, {}
    base = dict(n_estimators=400, subsample=0.80, colsample_bytree=0.75,
                gamma=0.15, reg_alpha=0.20, reg_lambda=1.5)

    keys = list(grid.keys())
    values = list(grid.values())

    for combo in product(*values):
        params = {**base, **dict(zip(keys, combo))}
        m = xgb.XGBClassifier(
            **params,
            objective='multi:softprob', num_class=num_class,
            eval_metric='mlogloss', early_stopping_rounds=20,
            random_state=42, verbosity=0,
        )
        m.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[sw_val] if sw_val is not None else None,
            verbose=False,
        )
        f1 = f1_score(m.predict(X_val), y_val, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_params = f1, params

    log.info(f"GridSearch fallback terminé | meilleur F1-macro val={best_f1:.4f}")
    log.info(f"  Meilleurs hyperparamètres : {best_params}")
    return best_params


# ============================Training XGBoost ==============================
def _apply_smote(X_tr, y_tr):

    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE

        counts = dict(zip(*np.unique(y_tr, return_counts=True)))
        n_na = counts.get(2, 0)
        # Cible : 60 % de NA pour limiter le sur-échantillonnage
        target = max(int(n_na * 0.60), 10)
        sampling = {}
        for cls, cnt in counts.items():
            if cls != 2 and cnt < target:
                k = min(5, cnt - 1)
                if k >= 1:
                    sampling[cls] = target
        if not sampling:
            log.info("SMOTE : toutes les classes minoritaires déjà suffisantes — ignoré")
            return X_tr, y_tr

        k_neighbors = min(5, min(counts[c] - 1 for c in sampling))
        smote = SMOTE(sampling_strategy=sampling, random_state=42, k_neighbors=k_neighbors)

        smt = SMOTETomek(smote=smote, random_state=42)
        X_res, y_res = smt.fit_resample(X_tr, y_tr)
        new_counts = dict(zip(*np.unique(y_res, return_counts=True)))
        log.info(f"SMOTETomek appliqué → distribution après : {new_counts}")
        return X_res, y_res
    except ImportError:
        log.warning("imbalanced-learn absent → SMOTE ignoré (pip install imbalanced-learn)")
        return X_tr, y_tr
    except Exception as e:
        log.warning(f"SMOTETomek échoué ({e}) → entraînement sans sur-échantillonnage")
        return X_tr, y_tr


def _calibrate_available_threshold(model, X_val, y_val, label_map):

    from sklearn.metrics import precision_recall_curve
    avail_enc = label_map.get('Available', 0)

    y_proba = model.predict_proba(X_val)
    p_avail = y_proba[:, avail_enc]
    y_true_bin = (y_val == avail_enc).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_true_bin, p_avail)

    THR_MIN, THR_MAX = 0.20, 0.60   # bornes élargies pour débloquer plus de Available

    PREC_MIN = 0.12  # seuil minimal de précision (accepter recall élevé)

    best_thr = THR_MIN
    best_f1 = 0.0
    for prec, rec, thr in zip(precision[:-1], recall[:-1], thresholds):
        if thr < THR_MIN or thr > THR_MAX:
            continue
        if prec < PREC_MIN:
            continue
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    if best_f1 == 0.0:
        log.warning(
            f"_calibrate_available_threshold : aucun seuil dans [{THR_MIN}, {THR_MAX}] "
            f"n'atteint precision>={PREC_MIN:.2f}. "
            f"Utilisation du seuil par défaut {THR_MIN:.3f}. "
            f"Envisager d'enrichir les features ou d'augmenter le volume Available."
        )

    actual_recall = float(recall[np.searchsorted(thresholds, best_thr, side='right') - 1]) \
        if len(thresholds) > 0 else 0.0
    actual_prec = float(precision[np.searchsorted(thresholds, best_thr, side='right') - 1]) \
        if len(thresholds) > 0 else 0.0

    log.info(
        f"Seuil Available calibré : {best_thr:.3f} "
        f"(precision={actual_prec:.3f}, recall={actual_recall:.3f}, "
        f"bornes=[{THR_MIN}, {THR_MAX}])"
    )
    return best_thr


def train_xgboost(df, label_map):
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    X = df[XGB_FEATURE_COLS].values
    y = df['label_enc'].values
    num_class = int(df['label_enc'].nunique())

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval,
    )


    X_tr, y_tr = _apply_smote(X_tr, y_tr)

    # Pondération des classes (en complément de SMOTE, pas à la place)
    sw_tr = compute_sample_weight('balanced', y_tr)
    sw_val = compute_sample_weight('balanced', y_val)

    log.info(f"Splits XGBoost : train={len(X_tr)} (après SMOTE) | val_ES={len(X_val)} | test={len(X_test)}")
    for cls, cnt in zip(*np.unique(y_tr, return_counts=True)):
        log.info(f"  classe {cls} → {cnt} exemples train")

    best_params = None
    if CONFIG.get('ENABLE_HYPERPARAMETER_TUNING', True):
        try:
            import optuna
            log.info(f"Optuna disponible → lancement du tuning bayésien ({CONFIG.get('TUNING_N_TRIALS', 50)} trials, CV {CONFIG.get('TUNING_CV_FOLDS', 3)}-fold)...")
            best_params = _tune_xgboost_optuna(
                X_tr, y_tr, X_val, y_val,
                num_class=num_class,
                sw_tr=sw_tr, sw_val=sw_val,
                n_trials=CONFIG.get('TUNING_N_TRIALS', 50),
            )
        except ImportError:
            log.warning("Optuna absent → fallback GridSearch partiel")
            try:
                best_params = _tune_xgboost_grid(X_tr, y_tr, X_val, y_val, num_class=num_class,
                                                 sw_tr=sw_tr, sw_val=sw_val)
            except Exception as e:
                log.warning(f"GridSearch échoué ({e}) → hyperparamètres par défaut")
        except Exception as e:
            log.warning(f"Optuna échoué ({e}) → hyperparamètres par défaut")

    if best_params is None:
        best_params = _default_xgb_params()
        log.info("Utilisation des hyperparamètres par défaut")

    model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softprob',
        num_class=num_class,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[sw_val.tolist()],
        verbose=False,
    )

    log.info(f"XGBoost entraîné — {model.best_iteration + 1} arbres — feature importances :")
    imp = sorted(zip(XGB_FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp_score in imp:
        log.info(f"    {feat:30s}  {imp_score:.4f}")

    avail_threshold = _calibrate_available_threshold(model, X_val, y_val, label_map)

    return model, X_test, y_test, avail_threshold

# ============================Évaluation XGBOOST ========================================================
def evaluate_model(model, X_test, y_test, label_map):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, classification_report, confusion_matrix,
    )
    from sklearn.utils.class_weight import compute_sample_weight

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    sw_test = compute_sample_weight('balanced', y_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0,
                    sample_weight=sw_test)

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except ValueError as e:
        log.warning(f"AUC non calculable : {e}")
        auc = float('nan')

    inv_map = {v: k for k, v in label_map.items()}
    target_names = [inv_map[i] for i in sorted(inv_map)]

    log.info("Evaluation XGBoost (test 20%, seed=42) :")
    log.info(f"  N={len(y_test)}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    log.info(f"  F1-macro={f1_mac:.4f}  F1-weighted(balanced)={f1_w:.4f}")
    if not np.isnan(auc):
        log.info(f"  AUC-ROC macro={auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"Matrice de confusion :\n{cm}")

    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True
    )
    for line in classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
    ).splitlines():
        if line.strip():
            log.info(f"  {line}")

    # Diagnostic actionnable par classe
    for cls_name in target_names:
        cls_metrics = report.get(cls_name, {})
        prec_cls = cls_metrics.get('precision', 0)
        rec_cls = cls_metrics.get('recall', 0)
        if cls_name == 'Available' and prec_cls < 0.60:
            log.warning(
                f"  ⚠ Available précision={prec_cls:.2f} < 0.60 — "
                f"des NA seront envoyés en campagne. "
                f"Compromis précision/recall acceptable pour EasyBulk : "
                f"viser recall > 0.30 plutôt que précision parfaite."
            )
        if cls_name == 'Suspected' and rec_cls < 0.40:
            log.warning(
                f"  ⚠ Suspected recall={rec_cls:.2f} < 0.40 — "
                f"la classe est sous-détectée. Vérifier la définition du label "
                f"et l'équilibre SMOTE."
            )
        if cls_name == 'NA' and prec_cls < 0.80:
            log.warning(
                f"  ⚠ NA précision={prec_cls:.2f} < 0.80 — "
                f"des contacts Available/Suspected sont exclus à tort."
            )

    # Métriques par classe pour le JSON
    per_class = {}
    for cls_name in target_names:
        m = report.get(cls_name, {})
        per_class[cls_name] = {
            'precision': round(m.get('precision', 0), 4),
            'recall': round(m.get('recall', 0), 4),
            'f1': round(m.get('f1-score', 0), 4),
            'support': int(m.get('support', 0)),
        }

    metrics = {
        'accuracy': round(float(acc), 4),
        'precision_macro': round(float(prec), 4),
        'recall_macro': round(float(rec), 4),
        'f1_macro': round(float(f1_mac), 4),
        'f1_weighted_balanced': round(float(f1_w), 4),
        'auc_roc_macro': round(float(auc), 4) if not np.isnan(auc) else None,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'test_split_size': 0.20,
        'test_split_seed': 42,
        'n_test_samples': len(y_test),
        'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        'evaluated_at': datetime.now().isoformat(),
    }
    with open(CONFIG['METRICS_JSON'], 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"Métriques → {CONFIG['METRICS_JSON']}")
    return metrics


# ==================================== RSF train ========================================
def train_rsf(df):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Horizons de prédiction : court terme (7j) et long terme (30j)
    T_COURT = 7.0
    T_LONG  = 30.0

    X     = df[RSF_FEATURE_COLS].values
    event = df['evenement_na'].astype(bool).values
    time  = df['duree_survie_jours'].clip(lower=1).values

    # Diagnostic durées — vérifier la dispersion pour RSF discriminant
    log.info(
        f"RSF duree_survie_jours — min={time.min():.0f}j | "
        f"p25={np.percentile(time,25):.0f}j | median={np.median(time):.0f}j | "
        f"p75={np.percentile(time,75):.0f}j | max={time.max():.0f}j | "
        f"std={time.std():.1f}j"
    )
    pct_sous_7  = 100.0 * (time < T_COURT).mean()
    pct_sous_30 = 100.0 * (time < T_LONG).mean()
    if time.std() < 5.0:
        log.warning(
            f"RSF : std(duree_survie_jours)={time.std():.1f}j très faible — "
            f"les prédictions p7 et p30 risquent d'être identiques. "
            f"Vérifier la correction de duree_survie_jours dans build_features()."
        )
    log.info(
        f"RSF : pct<{T_COURT:.0f}j={pct_sous_7:.1f}% | pct<{T_LONG:.0f}j={pct_sous_30:.1f}%"
    )

    y_rsf = Surv.from_arrays(event=event, time=time)

    # ── Diagnostic événements / censurés ─────────────────────────────────────
    durees_ev = time[event]
    durees_ce = time[~event]
    if len(durees_ev) > 0:
        log.info(
            f"RSF event=True  — n={len(durees_ev):,} | "
            f"min={durees_ev.min():.0f}j | median={np.median(durees_ev):.0f}j | "
            f"max={durees_ev.max():.0f}j | "
            f"pct<{T_COURT:.0f}j={100*(durees_ev < T_COURT).mean():.1f}% | "
            f"pct<{T_LONG:.0f}j={100*(durees_ev < T_LONG).mean():.1f}%"
        )
    else:
        log.warning("RSF : AUCUN evenement (event=True) — verifier evenement_na dans build_features()")
    if len(durees_ce) > 0:
        log.info(
            f"RSF event=False — n={len(durees_ce):,} | "
            f"min={durees_ce.min():.0f}j | median={np.median(durees_ce):.0f}j | "
            f"max={durees_ce.max():.0f}j"
        )

    # ── Split stratifie ───────────────────────────────────────────────────────
    n_events = int(event.sum())
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y_rsf, test_size=0.25, random_state=42,
        stratify=event if n_events >= 10 else None,
    )
    if n_events < 10:
        log.warning(f"RSF : seulement {n_events} evenements — stratification desactivee")

    # ── Scaling sans fuite ────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_ev_sc  = scaler.transform(X_eval)
    X_all_sc = scaler.transform(X)

    # ── Entrainement — paramètres adaptés pour limiter la RAM ────────────────
    # n_estimators réduit + max_samples pour éviter OOM sur grands datasets
    n_train = len(X_train)
    max_samp = min(n_train, 15_000)   # cap à 15k lignes par arbre → économise RAM
    rsf = RandomSurvivalForest(
        n_estimators=100,             # 400→100 : 4x moins de RAM à l'inférence
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        max_samples=max_samp,         # sous-échantillonnage par arbre
        random_state=42,
        n_jobs=1,                     # -1→1 : pas de parallélisme (évite OOM multi-thread)
    )
    rsf.fit(X_tr_sc, y_train)

    # C-index sur un sous-échantillon pour éviter OOM au score
    idx_sample = np.random.default_rng(42).choice(len(X_train), size=min(5000, len(X_train)), replace=False)
    c_tr = rsf.score(X_tr_sc[idx_sample], y_train[idx_sample])
    c_ev = rsf.score(X_ev_sc, y_eval)
    log.info(f"RSF — C-index train={c_tr:.4f} | eval={c_ev:.4f}")
    if c_tr - c_ev > 0.10:
        log.warning("RSF : gap C-index > 0.10 — surapprentissage possible")

    # ── Prédictions par batch pour éviter OOM (shape: n×n_unique_times) ───────
    BATCH_SIZE = 2000
    all_rows = []
    for start in range(0, len(X_all_sc), BATCH_SIZE):
        batch = X_all_sc[start:start + BATCH_SIZE]
        surv_batch = rsf.predict_survival_function(batch)
        for fn in surv_batch:
            s7  = float(np.interp(T_COURT, fn.x, fn.y, left=fn.y[0], right=fn.y[-1]))
            s30 = float(np.interp(T_LONG,  fn.x, fn.y, left=fn.y[0], right=fn.y[-1]))
            p7  = float(np.clip(1.0 - s7,  0.0, 1.0))
            p30 = max(float(np.clip(1.0 - s30, 0.0, 1.0)), p7)
            all_rows.append({'prob_na_7d': round(p7, 4), 'prob_na_30d': round(p30, 4)})
        if start % 10000 == 0 and start > 0:
            log.info(f"RSF prédictions : {start}/{len(X_all_sc)}")

    unique_times = rsf.unique_times_
    log.info(
        f"RSF unique_times — min={unique_times.min():.1f}j | "
        f"max={unique_times.max():.1f}j | n={len(unique_times)}"
    )
    if unique_times.max() < T_LONG:
        log.warning(
            f"RSF : unique_times max ({unique_times.max():.1f}j) < T_LONG ({T_LONG}j) "
            f"— p30 extrapole. Verifier duree_survie_jours dans build_features()."
        )

    # ── Diagnostic discriminabilite ───────────────────────────────────────────
    rows = all_rows   # alias — calculé dans le batch loop ci-dessus
    p7_arr  = np.array([r['prob_na_7d']  for r in rows])
    p30_arr = np.array([r['prob_na_30d'] for r in rows])
    distinct = int(np.sum(np.abs(p7_arr - p30_arr) > 0.01))
    log.info(
        f"RSF — {distinct}/{len(rows)} contacts p7!=p30 | "
        f"p7 : mean={p7_arr.mean():.3f} std={p7_arr.std():.3f} | "
        f"p30: mean={p30_arr.mean():.3f} std={p30_arr.std():.3f}"
    )
    if p7_arr.std() < 0.01:
        log.warning("RSF : std(p7) < 0.01 — predictions court terme non discriminantes")
    if distinct == 0:
        log.warning(
            f"RSF : p7 == p30 pour tous les contacts. "
            f"unique_times ne couvre pas [{T_COURT}j, {T_LONG}j]. "
            f"Verifier duree_survie_jours dans build_features()."
        )

    return rsf, pd.DataFrame(rows), scaler



# ========================== Contact Availability Score ===============================================================
def compute_scores(xgb_model, rsf_preds, X, avail_threshold=0.5, label_map=None, df_features=None):

    if label_map is None:
        label_map = {'Available': 0, 'Suspected': 1, 'NA': 2}
    idx_avail   = label_map.get('Available', 0)
    idx_suspect = label_map.get('Suspected', 1)
    idx_na      = label_map.get('NA', 2)
    n_classes   = len(label_map)

    xgb_proba = xgb_model.predict_proba(X)
    if xgb_proba.shape[1] != n_classes:
        log.warning(
            f"compute_scores : xgb_proba.shape[1]={xgb_proba.shape[1]} "
            f"≠ n_classes={n_classes} — vérifier label_map"
        )
        idx_avail   = min(idx_avail,   xgb_proba.shape[1] - 1)
        idx_suspect = min(idx_suspect, xgb_proba.shape[1] - 1)
        idx_na      = min(idx_na,      xgb_proba.shape[1] - 1)

    rsf_available = (
        rsf_preds is not None
        and isinstance(rsf_preds, pd.DataFrame)
        and len(rsf_preds) > 0
        and 'prob_na_7d' in rsf_preds.columns
    )

    if rsf_available:
        p7_vals  = rsf_preds['prob_na_7d'].values
        p30_vals = rsf_preds['prob_na_30d'].values
        mean_p30 = float(np.mean(p30_vals))
        rsf_discriminant = bool(
            np.std(p7_vals) > 0.01
            and np.mean(np.abs(p7_vals - p30_vals)) > 0.01
            and mean_p30 < 0.95
        )
        if mean_p30 >= 0.95:
            log.warning(
                f"RSF : mean_p30={mean_p30:.3f} >= 0.95 → p30 sature. "
                f"RSF desactive — decision basee sur XGBoost + heuristique temporelle."
            )
    else:
        rsf_discriminant = False

    if not rsf_discriminant:
        log.warning(
            "compute_scores : RSF non discriminant → décision basée sur XGBoost + heuristique temporelle."
        )

    jds_arr   = None
    srp_arr   = None
    t30_arr   = None
    ec_fin_arr = None
    if df_features is not None and not rsf_discriminant:
        if 'jours_depuis_succes' in df_features.columns:
            jds_arr = df_features['jours_depuis_succes'].values.astype(float)
        if 'score_recence_pondere' in df_features.columns:
            srp_arr = df_features['score_recence_pondere'].values.astype(float)
        if 'taux_succes_30j' in df_features.columns:
            t30_arr = df_features['taux_succes_30j'].values.astype(float)
        if 'echecs_consecutifs_fin' in df_features.columns:
            ec_fin_arr = df_features['echecs_consecutifs_fin'].values.astype(float)

    rows = []
    thr = SCORE_THRESHOLDS

    for i in range(len(X)):
        p_avail   = float(xgb_proba[i][idx_avail])
        p_suspect = float(xgb_proba[i][idx_suspect])
        p_na_xgb  = float(xgb_proba[i][idx_na])

        if rsf_discriminant and i < len(rsf_preds):
            p7  = float(rsf_preds.iloc[i]['prob_na_7d'])
            p30 = float(rsf_preds.iloc[i]['prob_na_30d'])

            na_risk_short  = 0.65 * p_na_xgb + 0.35 * p7
            na_risk_long   = 0.30 * p_na_xgb + 0.70 * p30
            na_risk_fusion = 0.50 * na_risk_short + 0.50 * na_risk_long

            if p_avail >= 0.85:
                na_risk_fusion *= (1.0 - 0.30 * p_avail)
            if p_na_xgb >= 0.80:
                na_risk_fusion = 0.70 * p_na_xgb + 0.30 * na_risk_fusion
            if abs(p_na_xgb - p7) > 0.40:
                na_risk_fusion = max(na_risk_fusion, 0.60 * p_na_xgb + 0.40 * p7)

            na_risk = float(np.clip(na_risk_fusion, 0.0, 1.0))
            p7_out, p30_out = p7, p30
        else:

            temporal_penalty = 0.0
            if jds_arr is not None:
                jds = jds_arr[i]
                temporal_penalty += float(np.clip(jds / 999.0, 0.0, 1.0)) * 0.20
            if srp_arr is not None:
                srp = srp_arr[i]
                temporal_penalty += float(np.clip(-srp / 5.0, 0.0, 1.0)) * 0.10
            if ec_fin_arr is not None:
                ec = ec_fin_arr[i]
                temporal_penalty += float(np.clip(ec / 6.0, 0.0, 1.0)) * 0.10

            na_risk = float(np.clip(p_na_xgb + temporal_penalty, 0.0, 1.0))
            p7_out = p_na_xgb
            p30_out = p_na_xgb

        score = round((1.0 - na_risk) * 100.0, 1)

        if p_avail >= avail_threshold and score >= thr['available']:
            decision, action = 'Available', 'Inclure dans la campagne'
        elif p_avail >= avail_threshold and score >= thr['suspected']:
            decision, action = 'Suspected', 'Surveiller — inclure avec prudence'
        elif score >= thr['suspected']:
            decision, action = 'Suspected', 'Surveiller — inclure avec prudence'
        else:
            decision, action = 'NA', 'Exclure de la campagne'

        rows.append({
            'availability_score': score,
            'p_available_%':  round(p_avail    * 100, 1),
            'p_suspect_%':    round(p_suspect   * 100, 1),
            'p_na_xgb_%':     round(p_na_xgb   * 100, 1),
            'p_na_rsf_7d_%':  round(p7_out      * 100, 1),
            'p_na_rsf_30d_%': round(p30_out     * 100, 1),
            'na_risk_fusion': round(na_risk, 4),
            'decision':       decision,
            'action':         action,
        })
    return pd.DataFrame(rows)


# ========================== Sauvegarde du modèle ======================================================
def save_model(xgb_model, rsf_model, rsf_scaler, label_map, eval_metrics=None, avail_threshold=0.5):
    xgb_model.save_model(CONFIG['MODEL_XGB'])
    FUSION_PARAMS = {
        'w_xgb_short': 0.65, 'w_rsf_short': 0.35,
        'w_xgb_long': 0.30, 'w_rsf_long': 0.70,   # p30 dominant (était 0.40/0.60)
        'w_short': 0.50, 'w_long': 0.50,            # équilibre short/long (était 0.65/0.35)
        'avail_confident_thr': 0.85, 'avail_reduction': 0.30,
        'na_confident_thr': 0.80, 'na_confident_w': 0.70,
        'divergence_thr': 0.40, 'divergence_w_xgb': 0.60, 'divergence_w_rsf': 0.40,
        'avail_threshold': avail_threshold,
    }
    artifacts = {
        'rsf_model': rsf_model,
        'rsf_scaler': rsf_scaler,
        'label_map': label_map,
        'xgb_feature_cols': XGB_FEATURE_COLS,
        'rsf_feature_cols': RSF_FEATURE_COLS,
        'rsf_extra_cols': RSF_EXTRA_COLS,
        'status_codes': {1: 'LIVRE', 2: 'NON_LIVRE'},
        'dlr_norm_mapping': {1: 1, 2: 2, 4: 2, 8: 2, 0: 2, 16: 2, 34: 2},
        'statuses_cleaned': [],
        'fusion_logic': 'temporal_split_no_leakage_v12',
        'score_thresholds': SCORE_THRESHOLDS,
        'fusion_params': FUSION_PARAMS,
        'avail_threshold': avail_threshold,
        'eval_metrics': eval_metrics or {},
        'trained_at': datetime.now().isoformat(),
        'version': '12.0.0',
    }
    joblib.dump(artifacts, CONFIG['MODEL_PKL'])
    size = os.path.getsize(CONFIG['MODEL_PKL']) // 1024
    log.info(f"Modèle → {CONFIG['MODEL_PKL']}  ({size} KB)")


def load_model_assets():
    return joblib.load(CONFIG['MODEL_PKL'])


# ========================== PIPELINE PRINCIPAL =====================================================================
def run_training():
    start = datetime.now()
    log.info("===== DEBUT TRAINING =====")
    try:
        if CONFIG['USE_MONGODB']:
            raw_gen = load_from_mongodb()

        valid_records, _ = filter_and_validate(raw_gen)
        contacts = normalise_statuses(valid_records)

        df_real = build_features(contacts)
        df_all, label_map = encode_labels(df_real)

        # ── XGBoost ──────────────────────────────────────────────────────────
        xgb_model, X_test, y_test, avail_threshold = train_xgboost(df_all, label_map)
        eval_metrics = evaluate_model(xgb_model, X_test, y_test, label_map)

        # ── RSF (entraîné sur df_real, prédictions sur df_real entier) ───────
        rsf_model, rsf_preds, rsf_scaler = train_rsf(df_real)

        # ── Scores finaux (sur tout df_real, RSF déjà aligné sur df_real) ────
        X_xgb = df_real[XGB_FEATURE_COLS].values
        # train_rsf retourne des prédictions sur l'ensemble complet (len == len(df_real))
        rsf_aligned = rsf_preds.reset_index(drop=True) if rsf_preds is not None else pd.DataFrame()
        scores_df = compute_scores(
            xgb_model, rsf_aligned, X_xgb,
            avail_threshold=avail_threshold,
            label_map=label_map,
            df_features=df_real.reset_index(drop=True),
        )

        result_df = pd.concat([
            df_real[['msisdn', 'label']].reset_index(drop=True),
            scores_df.reset_index(drop=True),
        ], axis=1)
        result_df.to_csv(CONFIG['PREDICTIONS_CSV'], index=False)

        # Distribution finale avec noms de classes
        dist = result_df['decision'].value_counts().to_dict()
        total = len(result_df)
        log.info(f"Distribution finale ({total} contacts) :")
        for lbl, cnt in dist.items():
            log.info(f"  {lbl:12s} → {cnt:5d}  ({cnt / total * 100:.1f}%)")

        # Cohérence label XGBoost vs décision finale
        match = (result_df['label'] == result_df['decision']).mean()
        log.info(f"Cohérence label/décision : {match * 100:.1f}%")

        save_model(xgb_model, rsf_model, rsf_scaler, label_map, eval_metrics, avail_threshold=avail_threshold)
        log.info(f"===== TRAINING TERMINE ({(datetime.now() - start).total_seconds():.1f}s) =====")

    except Exception as e:
        log.error(f"ERREUR TRAINING : {e}", exc_info=True)


# =============================== CRON JOB : Inférence quotidienne ==================================================
def run_daily_inference():
    log.info("=== CronJob INFERENCE QUOTIDIENNE ===")
    if not os.path.exists(CONFIG['MODEL_PKL']) or not os.path.exists(CONFIG['MODEL_XGB']):
        log.warning("Modèle introuvable — lancez d'abord le training.")
        return

    start = datetime.now()
    try:
        artifacts = load_model_assets()
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(CONFIG['MODEL_XGB'])
        rsf_model = artifacts['rsf_model']
        rsf_scaler = artifacts['rsf_scaler']

        # Colonnes sauvegardées lors du training (compatibilité)
        xgb_cols = artifacts.get('xgb_feature_cols', XGB_FEATURE_COLS)
        rsf_cols = artifacts.get('rsf_feature_cols', RSF_FEATURE_COLS)

        if CONFIG['USE_MONGODB']:
            raw_gen = load_from_mongodb()

        valid_records, _ = filter_and_validate(raw_gen)
        contacts = normalise_statuses(valid_records)
        df_real = build_features(contacts)

        if df_real.empty:
            log.warning("Aucune donnée pour l'inférence")
            return

        X_xgb = df_real[xgb_cols].values
        X_rsf = df_real[rsf_cols].values

        # RSF : prédictions avec le scaler sauvegardé
        if rsf_model is not None and rsf_scaler is not None:
            X_scaled = rsf_scaler.transform(X_rsf)
            surv_funcs = rsf_model.predict_survival_function(X_scaled)
            rsf_rows = []
            for fn in surv_funcs:
                times = fn.x
                surv  = fn.y
                s7  = float(np.interp(7.0,  times, surv, left=surv[0], right=surv[-1]))
                s30 = float(np.interp(30.0, times, surv, left=surv[0], right=surv[-1]))
                p7  = float(np.clip(1.0 - s7,  0.0, 1.0))
                p30 = float(np.clip(1.0 - s30, 0.0, 1.0))
                p30 = max(p30, p7)  # monotonie
                rsf_rows.append({'prob_na_7d': round(p7, 4), 'prob_na_30d': round(p30, 4)})
            rsf_df = pd.DataFrame(rsf_rows)
        else:
            _, rsf_df, _ = train_rsf(df_real)

        scores_df = compute_scores(
            xgb_model,
            rsf_df.iloc[:len(df_real)].reset_index(drop=True),
            X_xgb,
            avail_threshold=artifacts.get('avail_threshold', 0.5),
            label_map=artifacts.get('label_map', None),
            df_features=df_real.reset_index(drop=True),
        )

        result_df = pd.concat([
            df_real[['msisdn', 'label']].reset_index(drop=True),
            scores_df.reset_index(drop=True),
        ], axis=1)
        result_df.to_csv(CONFIG['PREDICTIONS_CSV'], index=False)

        dist = result_df['decision'].value_counts().to_dict()
        total = len(result_df)
        log.info(f"Inférence terminée — {total} contacts")
        for lbl, cnt in dist.items():
            log.info(f"  {lbl:12s} -> {cnt:5d}  ({cnt / total * 100:.1f}%)")
        log.info(f"Durée : {(datetime.now() - start).total_seconds():.1f}s")

    except Exception as e:
        log.error(f"ERREUR inférence : {e}", exc_info=True)


# ========================= PLANIFICATION NOCTURNE ==================================================================
def _seconds_until(hhmm):
    now = datetime.now()
    h, m = map(int, hhmm.split(':'))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _scheduler_loop():
    schedule_time = CONFIG['SCHEDULE_TIME']
    retrain_interval = CONFIG.get('RETRAIN_INTERVAL_DAYS', 14)
    last_retrain = None

    while True:
        wait = _seconds_until(schedule_time)
        h, m = divmod(int(wait // 60), 60)
        now = datetime.now()
        need_retrain = last_retrain is None or (now - last_retrain).days >= retrain_interval

        if need_retrain:
            log.info(f"Prochain RE-TRAINING dans {h}h {m}min")
        else:
            log.info(
                f"Prochain cycle dans {h}h {m}min (inférence — re-training dans ~{retrain_interval - (now - last_retrain).days}j)")

        time.sleep(wait)
        if need_retrain:
            log.info("=== CronJob RE-TRAINING ===")
            run_training()
            last_retrain = datetime.now()
        else:
            run_daily_inference()


def start_scheduler():
    if CONFIG['RUN_NOW']:
        log.info("Exécution immédiate au démarrage...")
        run_training()
    t = Thread(target=_scheduler_loop, daemon=True, name='nightly-training')
    t.start()
    log.info(f"Scheduler actif — training planifié à {CONFIG['SCHEDULE_TIME']}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log.info("Scheduler arrêté.")


# ==============p.p=========================
if __name__ == '__main__':
    start_scheduler()