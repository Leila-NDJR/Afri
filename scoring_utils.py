#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fonctions partagées par les notebooks de paliers (M0, M1, M2, M3, ...).

Un seul endroit pour le pipeline de modélisation, la recherche du meilleur
C, la lecture des coefficients, les métriques (AUC/Gini/KS) et le test de
DeLong, pour éviter de redéfinir ces fonctions dans chaque notebook.
"""

import re

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================ variables comportementales ==================

def construire_variables_comportementales(tx, clients):
    """Agrège les transactions en une ligne par client (variables agnostiques au libellé)."""
    tx = tx.copy()
    tx["abs_m"] = tx.montant.abs()
    tx["jour"] = pd.to_datetime(tx.date).dt.dayofyear
    g = tx.groupby("client_id")

    f = pd.DataFrame({
        "nb_tx":           g.size(),
        "pct_debits":      g.apply(lambda d: (d.montant < 0).mean(), include_groups=False),
        "inflow":          g.apply(lambda d: d.montant[d.montant > 0].sum(), include_groups=False),
        "outflow":         g.apply(lambda d: -d.montant[d.montant < 0].sum(), include_groups=False),
        "mean_abs":        g.abs_m.mean(),
        "std_abs":         g.abs_m.std().fillna(0),
        "max_abs":         g.abs_m.max(),
        "nb_jours_actifs": g.jour.nunique(),
    })
    f["net_flow"] = f.inflow - f.outflow
    f["cv_abs"] = (f.std_abs / f.mean_abs.replace(0, np.nan)).fillna(0)
    f["tx_par_jour"] = f.nb_tx / f.nb_jours_actifs.replace(0, np.nan)

    f = f.merge(clients[["client_id", "revenu_declare"]].set_index("client_id"),
                left_index=True, right_index=True)
    f["ecart_revenu"] = (f.revenu_declare - f.inflow / 3.0) / f.revenu_declare.replace(0, np.nan)
    return f.drop(columns=["revenu_declare"]).reset_index().rename(columns={"index": "client_id"})


# ============================ métriques ==================================

def ks_stat(y, p):
    """Statistique de Kolmogorov-Smirnov : écart max entre les CDF des deux classes."""
    ordre = np.argsort(p)
    y = np.asarray(y)[ordre]
    pos = np.cumsum(y) / max(y.sum(), 1)
    neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def evaluer(nom, y_true, p):
    """Affiche AUC, Gini et KS pour des probabilités déjà prédites."""
    auc = roc_auc_score(y_true, p)
    ks = ks_stat(y_true, p)
    print(f"{nom:4} | AUC {auc:.3f} | Gini {2 * auc - 1:.3f} | KS {ks:.3f}")
    return {"nom": nom, "auc": auc, "gini": 2 * auc - 1, "ks": ks}


# ---- DeLong (Sun & Xu 2014) pour comparer deux AUC corrélées ----

def _midrank(x):
    """Rangs moyens (gère les ex-aequo) pour le calcul rapide de DeLong."""
    ordre = np.argsort(x)
    z = x[ordre]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    t2 = np.empty(n)
    t2[ordre] = t
    return t2


def _fast_delong(preds, m):
    """Implémentation rapide de DeLong et al. (1988) : AUC et matrice de covariance."""
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, cov


def delong_test(y, p1, p2):
    """p-value bilatérale H0 : AUC(p1) == AUC(p2), sur le même jeu de test."""
    from scipy.stats import norm

    y = np.asarray(y)
    ordre = np.r_[np.where(y == 1)[0], np.where(y == 0)[0]]
    m = int((y == 1).sum())
    preds = np.vstack([np.asarray(p1)[ordre], np.asarray(p2)[ordre]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs, float(2 * norm.sf(abs(z)))


def chaine_delong(y, resultats):
    """Test de DeLong entre chaque palier et le précédent, dans l'ordre du dict `resultats`.

    `resultats` : dict ordonné {nom_palier: probabilites_predites}, ex.
    {"M1": p1, "M2": p2, "M3": p3}. Affiche une ligne par comparaison consécutive.
    """
    noms = list(resultats)
    for nom_avant, nom_apres in zip(noms, noms[1:]):
        aucs, pval = delong_test(y, resultats[nom_avant], resultats[nom_apres])
        conclusion = "significatif" if pval < 0.05 else "non significatif"
        print(f"DeLong {nom_avant}->{nom_apres} : AUC {aucs[0]:.3f} -> {aucs[1]:.3f} | "
              f"p = {pval:.2e} ({conclusion})")


# ============================ pipeline ====================================

def construire_pipeline(num_cols, cat_cols, texte_col=None, seed=42):
    """Pipeline standard d'un palier : mise à l'échelle + one-hot [+ texte] + régression logistique.

    Si `texte_col` est fourni, ajoute une branche TF-IDF de n-grammes de
    caractères (robuste aux fautes/troncatures) réduite par SVD à 30
    dimensions denses — c'est la seule différence entre un palier M0/M1/M2
    et un palier M3+.
    """
    etapes = [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
    if texte_col is not None:
        pipeline_texte = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=4000)),
            ("svd", TruncatedSVD(n_components=30, random_state=seed)),
        ])
        etapes.append(("txt", pipeline_texte, texte_col))

    pretraitement = ColumnTransformer(etapes)
    return Pipeline([
        ("pre", pretraitement),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


def construire_pipeline_arbre(num_cols, cat_cols, y=None, modele="xgboost", texte_col=None, seed=42):
    """Pipeline non-linéaire de référence (XGBoost ou Random Forest), même prétraitement que `construire_pipeline`.

    Sert de comparaison à la régression logistique (`construire_pipeline`) sur le
    *même* jeu de variables (`num_cols`/`cat_cols`/`texte_col`) : seule l'étape
    finale (`clf`) change, la mise à l'échelle / one-hot / TF-IDF+SVD sont
    identiques, pour isoler l'effet de la non-linéarité.

    Gestion du déséquilibre de classes (équivalent à `class_weight="balanced"`
    de la régression logistique) :
    - `modele="xgboost"` : `scale_pos_weight` = ratio négatifs/positifs du
      train, calculé à partir de `y` (le train, à passer explicitement — pas
      de `class_weight` natif pour XGBoost) ;
    - `modele="random_forest"` : `class_weight="balanced"` nativement.
    """
    etapes = [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
    if texte_col is not None:
        pipeline_texte = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=4000)),
            ("svd", TruncatedSVD(n_components=30, random_state=seed)),
        ])
        etapes.append(("txt", pipeline_texte, texte_col))
    pretraitement = ColumnTransformer(etapes)

    if modele == "xgboost":
        from xgboost import XGBClassifier

        scale_pos_weight = 1.0
        if y is not None:
            y = np.asarray(y)
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=scale_pos_weight, eval_metric="auc",
            random_state=seed, n_jobs=-1,
        )
    elif modele == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError(f"modele inconnu : {modele!r} (attendu 'xgboost' ou 'random_forest')")

    return Pipeline([("pre", pretraitement), ("clf", clf)])


def bootstrap_auc_ci(y_true, p, n_boot=2000, seed=42, ci=0.95):
    """IC bootstrap (percentile) simple sur l'AUC d'un seul modèle.

    Rééchantillonne le jeu de test (lignes, avec remise) `n_boot` fois et
    recalcule l'AUC à chaque fois — ne suppose pas, contrairement à DeLong,
    que les deux modèles comparés partagent les mêmes erreurs corrélées :
    une vérification indépendante, plus conservatrice, de la significativité
    d'un écart d'AUC entre deux modèles évalués sur le même test.
    Retourne (auc_observee, (borne_basse, borne_haute)).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    n = len(y_true)

    aucs = []
    tentatives = 0
    while len(aucs) < n_boot and tentatives < n_boot * 10:
        tentatives += 1
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.min() == yb.max():  # rééchantillon dégénéré (une seule classe) : on l'ignore
            continue
        aucs.append(roc_auc_score(yb, p[idx]))
    aucs = np.array(aucs)

    alpha = (1 - ci) / 2
    return float(roc_auc_score(y_true, p)), (float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha)))


def rechercher_meilleur_C(pipeline, X, y, grille_C, cv=5, seed=42):
    """Recherche du meilleur C (régularisation) par validation croisée sur l'AUC.

    Retourne (meilleur_estimateur_refit_sur_tout_X, tableau C -> AUC moyen ± écart-type).
    """
    decoupage = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    recherche = GridSearchCV(
        pipeline,
        param_grid={"clf__C": grille_C},
        scoring="roc_auc",
        cv=decoupage,
        refit=True,
    )
    recherche.fit(X, y)
    resultats = pd.DataFrame({
        "C": recherche.cv_results_["param_clf__C"],
        "auc_cv_moyen": recherche.cv_results_["mean_test_score"],
        "auc_cv_ecart_type": recherche.cv_results_["std_test_score"],
    }).sort_values("C").reset_index(drop=True)
    return recherche.best_estimator_, resultats


def extraire_coefficients(pipeline, num_cols, cat_cols):
    """Table variable -> coefficient du LogisticRegression entraîné, triée par poids absolu.

    Les coefficients portent sur des variables standardisées (num), one-hot (cat) ou,
    pour un palier avec texte (voir `construire_pipeline`), sur les 30 dimensions
    denses issues du TF-IDF+SVD (nommées `texte_svd_0..29` : un poids visible, mais
    pas interprétable mot à mot comme les variables numériques ou catégorielles).
    """
    noms = [n.split("__", 1)[-1] for n in pipeline.named_steps["pre"].get_feature_names_out()]
    noms = [re.sub(r"truncatedsvd(\d+)", r"texte_svd_\1", n) for n in noms]
    coefs = pipeline.named_steps["clf"].coef_[0]
    table = pd.DataFrame({"variable": noms, "coefficient": coefs})
    table["poids_absolu"] = table.coefficient.abs()
    return table.sort_values("poids_absolu", ascending=False).drop(columns="poids_absolu").reset_index(drop=True)


# ============================ audit d'équité ==============================

def resumer_decision_equite(df, approuve, group_col, gt_true_col=None):
    """Audit d'équité à partir d'une décision d'approbation déjà calculée (0/1).

    Retourne (taux_approbation_par_groupe, ratio_4_5, refus_des_vrais_bons_ou_None).
    Ratio < 0,8 signale un *disparate impact* (règle des 4/5). Si `gt_true_col`
    est fourni, calcule aussi la part des clients qui n'auraient réellement
    pas fait défaut (`gt_true_col == 0`) mais que la décision refuse quand
    même — l'injustice injectée par le biais d'étiquette, à distinguer d'un
    écart de risque réel. Séparée de `audit_equite` pour pouvoir auditer une
    décision issue de seuils différents par groupe (voir `appliquer_seuils_groupe`).
    """
    d = df.assign(approuve=np.asarray(approuve))
    taux = d.groupby(group_col).approuve.mean()
    ratio = taux.min() / taux.max() if taux.max() > 0 else 0.0

    print(f"taux d'approbation par {group_col} : {taux.round(3).to_dict()}")
    print(f"ratio (règle des 4/5) : {ratio:.2f} -> "
          f"{'disparate impact' if ratio < 0.8 else 'pas de signal au seuil des 4/5'}")

    refus_bons = None
    if gt_true_col is not None:
        bons = d[d[gt_true_col] == 0]
        refus_bons = bons.groupby(group_col).apply(lambda x: (x.approuve == 0).mean(), include_groups=False)
        print(f"refus des vrais bons par {group_col} : {refus_bons.round(3).to_dict()}")

    return taux, ratio, refus_bons


def audit_equite(df, p, group_col, gt_true_col=None):
    """Audit d'équité au seuil d'approbation médian unique de `p` (voir `resumer_decision_equite`)."""
    seuil = np.median(p)
    return resumer_decision_equite(df, p < seuil, group_col, gt_true_col)


# ============================ correction d'équité ==========================

def seuils_egalite_opportunite(df, p, group_col, gt_true_col):
    """Seuils d'approbation par groupe qui égalisent le taux de refus des vrais bons.

    À calculer sur le TRAIN uniquement, puis à appliquer tel quel sur le test
    via `appliquer_seuils_groupe` (jamais recalculer sur le test : fuite).
    Le groupe le mieux traité au seuil médian global (celui qui refuse le
    moins ses vrais bons) sert de cible ; pour les autres groupes, on
    cherche sur une grille de quantiles de `p` le seuil qui s'en rapproche
    le plus — une version simplifiée, par recherche sur grille, de
    l'*equal opportunity* (Hardt, Price & Srebro, 2016).
    """
    seuil_global = np.median(p)
    d = df.assign(p=np.asarray(p), bon=(df[gt_true_col] == 0))

    def taux_refus(seuil, sous):
        bons = sous[sous.bon]
        return float((bons.p >= seuil).mean()) if len(bons) else 0.0

    taux_par_groupe = {g: taux_refus(seuil_global, sous) for g, sous in d.groupby(group_col)}
    groupe_cible = min(taux_par_groupe, key=taux_par_groupe.get)
    taux_cible = taux_par_groupe[groupe_cible]

    grille = np.quantile(d.p, np.linspace(0.01, 0.99, 99))
    seuils = {}
    for g, sous in d.groupby(group_col):
        if g == groupe_cible:
            seuils[g] = seuil_global
            continue
        ecarts = [abs(taux_refus(s, sous) - taux_cible) for s in grille]
        seuils[g] = float(grille[int(np.argmin(ecarts))])
    return seuils


def appliquer_seuils_groupe(df, p, group_col, seuils):
    """Décision d'approbation (0/1) à partir de seuils différents par groupe."""
    seuil_par_ligne = df[group_col].map(seuils)
    return (np.asarray(p) < seuil_par_ligne).astype(int)


# ============================ explicabilité (SHAP) ========================

def calculer_shap(pipeline, X_fond, X_cible, echantillon_fond=200, seed=42):
    """Valeurs SHAP (`LinearExplainer`) sur la sortie log-odds du `LogisticRegression` entraîné.

    Exact pour un modèle linéaire (pas d'approximation Kernel/Permutation) : chaque
    valeur SHAP est la contribution de la variable à l'écart entre le score log-odds
    du client et une valeur de base (`expected_value`), sur l'espace déjà transformé
    par `pipeline.named_steps["pre"]` (mêmes noms de variable que `extraire_coefficients`,
    y compris le renommage des composantes SVD texte en `texte_svd_0..N`).

    `X_fond` sert de distribution de référence (sous-échantillonnée à `echantillon_fond`
    lignes si plus grande, typiquement le train) ; les valeurs SHAP sont calculées pour
    chaque ligne de `X_cible` (typiquement le test). Retourne (DataFrame [n_cible x
    n_variables], valeur_base).
    """
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    fond = X_fond.sample(n=min(echantillon_fond, len(X_fond)), random_state=seed)
    fond_transforme = pre.transform(fond)
    cible_transforme = pre.transform(X_cible)
    if hasattr(fond_transforme, "toarray"):
        fond_transforme = fond_transforme.toarray()
    if hasattr(cible_transforme, "toarray"):
        cible_transforme = cible_transforme.toarray()

    explainer = shap.LinearExplainer(clf, fond_transforme)
    valeurs = explainer.shap_values(cible_transforme)

    noms = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    noms = [re.sub(r"truncatedsvd(\d+)", r"texte_svd_\1", n) for n in noms]
    table = pd.DataFrame(valeurs, columns=noms, index=X_cible.index)
    return table, float(explainer.expected_value)


def shap_par_sousgroupe(valeurs_shap, meta, group_col, top_n=15):
    """Importance moyenne (`|SHAP|`) par variable, comparée entre sous-groupes de `group_col`.

    `meta` doit être indexable comme `valeurs_shap` (même index que `X_cible` passé à
    `calculer_shap`) et contenir `group_col`. Retourne un tableau variable x sous-groupe
    (moyenne de `|SHAP|` dans chaque sous-groupe), limité aux `top_n` variables les plus
    importantes au global — pour repérer, au-delà du classement moyen déjà donné par
    `extraire_coefficients`, une variable qui pèse très différemment d'un sous-groupe à
    l'autre (utile à l'explicabilité réglementaire par profil, pas seulement en moyenne).
    """
    abs_shap = valeurs_shap.abs()
    ordre_global = abs_shap.mean().sort_values(ascending=False).head(top_n).index

    table = abs_shap[ordre_global].copy()
    table["_groupe"] = meta.loc[valeurs_shap.index, group_col].values
    return table.groupby("_groupe").mean().T.reindex(ordre_global)


# ============================ gain financier / provisionnement COBAC (H5) ==

def defauts_evites_a_taux(y_true, p, taux_approbation):
    """Nombre de défauts parmi les dossiers approuvés à `taux_approbation` (0-1).

    Décision d'octroi hypothétique par seuil de score, dans le même esprit que
    `audit_equite` : on approuve les `taux_approbation` % de dossiers les moins
    risqués selon `p` (probabilité de défaut, donc score croissant = plus sûr
    d'abord). Retourne (n_approuves, n_defauts_parmi_approuves).
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    n_approuves = int(round(taux_approbation * len(y_true)))
    approuves = np.argsort(p)[:n_approuves]
    return n_approuves, int(y_true[approuves].sum())


def grille_gain_provisionnement(y_true, p_ancien, p_nouveau, grille_taux_approbation,
                                 montant_moyen_credit, taux_provisionnement, volume_annuel):
    """Économie de provisionnement estimée en substituant `p_nouveau` à `p_ancien`, à volume d'octroi égal.

    Hypothèse explicite (voir le notebook `gain_financier.ipynb`) : décision d'octroi
    hypothétique par seuil de score sur le portefeuille déjà observé, faute de données
    de refus réelles. Pour chaque taux d'approbation de la grille, le taux de défauts
    évités par dossier approuvé (mesuré sur le jeu de test) est mis à l'échelle du
    `volume_annuel` de dossiers, puis valorisé par `montant_moyen_credit *
    taux_provisionnement` (les deux étant des hypothèses paramétriques, pas des valeurs
    mesurées). Retourne un DataFrame, une ligne par taux d'approbation de la grille.
    """
    lignes = []
    for taux in grille_taux_approbation:
        n_a, def_a = defauts_evites_a_taux(y_true, p_ancien, taux)
        _, def_n = defauts_evites_a_taux(y_true, p_nouveau, taux)
        taux_defauts_evites = (def_a - def_n) / n_a if n_a else 0.0
        defauts_evites_annuels = taux_defauts_evites * volume_annuel
        lignes.append({
            "taux_approbation": taux,
            "defauts_evites_test": def_a - def_n,
            "defauts_evites_annuels_estimes": defauts_evites_annuels,
            "gain_fcfa_an": defauts_evites_annuels * montant_moyen_credit * taux_provisionnement,
        })
    return pd.DataFrame(lignes)


def bootstrap_gain_provisionnement(y_true, p_ancien, p_nouveau, taux_approbation,
                                    montant_moyen_credit, taux_provisionnement, volume_annuel,
                                    n_boot=2000, seed=42, ci=0.95):
    """IC bootstrap (percentile) sur le gain FCFA/an, à un taux d'approbation fixé.

    Rééchantillonne le jeu de test (lignes, avec remise) `n_boot` fois et rejoue
    `grille_gain_provisionnement` sur une grille à un seul taux, pour quantifier
    l'incertitude d'échantillonnage sur le nombre de défauts évités — dans le même
    esprit que le test de DeLong pour les AUC, mais ici pour le gain financier dérivé.
    Retourne (gain_moyen, (borne_basse, borne_haute)).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p_ancien = np.asarray(p_ancien)
    p_nouveau = np.asarray(p_nouveau)
    n = len(y_true)

    gains = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        n_a, def_a = defauts_evites_a_taux(y_true[idx], p_ancien[idx], taux_approbation)
        _, def_n = defauts_evites_a_taux(y_true[idx], p_nouveau[idx], taux_approbation)
        taux_evites = (def_a - def_n) / n_a if n_a else 0.0
        gains[i] = taux_evites * volume_annuel * montant_moyen_credit * taux_provisionnement

    alpha = (1 - ci) / 2
    return float(gains.mean()), (float(np.quantile(gains, alpha)), float(np.quantile(gains, 1 - alpha)))


def poids_reponderation(y, groupes):
    """Poids d'entraînement de Kamiran & Calders (2012) par (groupe, label).

    w(g,y) = P(g)·P(y) / P(g,y) — neutralise la corrélation groupe↔label
    avant l'entraînement (repondération des données, pas du modèle). À
    passer en `sample_weight` lors du `.fit()`.
    """
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    g = pd.Series(np.asarray(groupes)).reset_index(drop=True)
    n = len(y)
    poids = pd.Series(1.0, index=y.index)
    for (gv, yv), sous in pd.DataFrame({"g": g, "y": y}).groupby(["g", "y"]):
        p_g = (g == gv).mean()
        p_y = (y == yv).mean()
        p_gy = len(sous) / n
        poids.loc[sous.index] = (p_g * p_y) / p_gy
    return poids.values
