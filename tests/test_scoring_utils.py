#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests unitaires de scoring_utils.py (pipeline, métriques, DeLong, équité, arbres).

Un cas nominal + au moins un cas limite par fonction publique majeure, sur un
petit jeu de données synthétique rapide à entraîner (voir conftest.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring_utils import (
    appliquer_seuils_groupe,
    audit_equite,
    bootstrap_auc_ci,
    bootstrap_gain_provisionnement,
    calculer_shap,
    chaine_delong,
    construire_pipeline,
    construire_pipeline_arbre,
    construire_variables_comportementales,
    defauts_evites_a_taux,
    delong_test,
    evaluer,
    extraire_coefficients,
    grille_gain_provisionnement,
    ks_stat,
    poids_reponderation,
    rechercher_meilleur_C,
    resumer_decision_equite,
    seuils_egalite_opportunite,
    shap_par_sousgroupe,
)


# ============================ construire_variables_comportementales ========

class TestConstruireVariablesComportementales:
    def test_nominal(self):
        clients = pd.DataFrame({"client_id": [1], "revenu_declare": [100_000.0]})
        tx = pd.DataFrame({
            "client_id": [1, 1, 1],
            "montant": [50_000.0, -20_000.0, -10_000.0],
            "date": ["2024-01-01", "2024-01-02", "2024-02-15"],
        })
        f = construire_variables_comportementales(tx, clients)
        row = f.set_index("client_id").loc[1]

        assert row["nb_tx"] == 3
        assert row["inflow"] == pytest.approx(50_000.0)
        assert row["outflow"] == pytest.approx(30_000.0)
        assert row["net_flow"] == pytest.approx(20_000.0)
        assert row["pct_debits"] == pytest.approx(2 / 3)
        assert "ecart_revenu" in f.columns
        assert "revenu_declare" not in f.columns  # colonne de travail retirée

    def test_cas_limite_montant_constant_pas_de_division_par_zero(self):
        # une seule transaction -> std_abs NaN comblé à 0, cv_abs ne doit pas être NaN/inf
        clients = pd.DataFrame({"client_id": [1], "revenu_declare": [50_000.0]})
        tx = pd.DataFrame({
            "client_id": [1],
            "montant": [0.0],
            "date": ["2024-01-01"],
        })
        f = construire_variables_comportementales(tx, clients)
        row = f.set_index("client_id").loc[1]
        assert row["mean_abs"] == 0.0
        assert not np.isnan(row["cv_abs"])
        assert not np.isinf(row["cv_abs"])


# ============================ métriques =====================================

class TestKsStat:
    def test_nominal_separation_parfaite(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert ks_stat(y, p) == pytest.approx(1.0)

    def test_cas_limite_une_seule_classe_ne_leve_pas(self):
        y = np.zeros(5, dtype=int)
        p = np.array([0.1, 0.4, 0.3, 0.9, 0.5])
        resultat = ks_stat(y, p)  # ne doit pas lever, doit rester une valeur finie
        assert np.isfinite(resultat)


class TestEvaluer:
    def test_nominal(self, capsys):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        resultat = evaluer("TEST", y, p)
        assert resultat["nom"] == "TEST"
        assert resultat["auc"] == pytest.approx(roc_auc_score(y, p))
        assert resultat["gini"] == pytest.approx(2 * resultat["auc"] - 1)
        assert "TEST" in capsys.readouterr().out

    def test_cas_limite_predictions_constantes(self):
        # p constant -> AUC dégénérée (0.5 par convention sklearn), ne doit pas lever
        y = np.array([0, 1, 0, 1])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        resultat = evaluer("CONST", y, p)
        assert resultat["auc"] == pytest.approx(0.5)


class TestDelongTest:
    def test_nominal_predictions_identiques(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        aucs, pval = delong_test(y, p, p)
        assert aucs[0] == pytest.approx(aucs[1])
        assert pval == pytest.approx(1.0)

    def test_cas_limite_modeles_nettement_differents(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p_bon = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        p_mauvais = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])  # inversé
        aucs, pval = delong_test(y, p_bon, p_mauvais)
        assert aucs[0] > aucs[1]
        assert 0.0 <= pval <= 1.0


def test_chaine_delong_affiche_chaque_comparaison(capsys):
    y = np.array([0, 0, 0, 1, 1, 1])
    p1 = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    p2 = np.array([0.1, 0.3, 0.5, 0.6, 0.75, 0.95])
    chaine_delong(y, {"A": p1, "B": p2})
    sortie = capsys.readouterr().out
    assert "DeLong A->B" in sortie


# ============================ pipeline ======================================

class TestConstruirePipeline:
    def test_nominal(self, petit_jeu_donnees):
        X, y = petit_jeu_donnees
        pipe = construire_pipeline(["x1", "x2"], ["cat"], seed=0)
        pipe.fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        assert p.shape == (len(X),)
        assert roc_auc_score(y, p) > 0.7  # signal fort et appris

    def test_avec_colonne_texte(self, petit_jeu_donnees_texte):
        X, y = petit_jeu_donnees_texte
        pipe = construire_pipeline(["x1", "x2"], ["cat"], texte_col="texte", seed=0)
        pipe.fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        assert p.shape == (len(X),)


class TestConstruirePipelineArbre:
    def test_nominal_xgboost(self, petit_jeu_donnees):
        X, y = petit_jeu_donnees
        pipe = construire_pipeline_arbre(["x1", "x2"], ["cat"], y=y, modele="xgboost", seed=0)
        pipe.fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        assert p.shape == (len(X),)
        assert roc_auc_score(y, p) > 0.7

    def test_nominal_random_forest(self, petit_jeu_donnees):
        X, y = petit_jeu_donnees
        pipe = construire_pipeline_arbre(["x1", "x2"], ["cat"], modele="random_forest", seed=0)
        pipe.fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        assert p.shape == (len(X),)

    def test_scale_pos_weight_calcule_depuis_y(self, petit_jeu_donnees):
        X, y = petit_jeu_donnees
        pipe = construire_pipeline_arbre(["x1", "x2"], ["cat"], y=y, modele="xgboost", seed=0)
        attendu = float((y == 0).sum() / (y == 1).sum())
        assert pipe.named_steps["clf"].scale_pos_weight == pytest.approx(attendu)

    def test_cas_limite_modele_inconnu_leve_value_error(self):
        with pytest.raises(ValueError):
            construire_pipeline_arbre(["x1", "x2"], ["cat"], modele="modele_qui_n_existe_pas")


class TestRechercherMeilleurC:
    def test_nominal(self, petit_jeu_donnees):
        X, y = petit_jeu_donnees
        pipe = construire_pipeline(["x1", "x2"], ["cat"], seed=0)
        modele, cv = rechercher_meilleur_C(pipe, X, y, [0.1, 1.0], cv=2, seed=0)
        assert set(cv.columns) == {"C", "auc_cv_moyen", "auc_cv_ecart_type"}
        assert len(cv) == 2
        p = modele.predict_proba(X)[:, 1]
        assert roc_auc_score(y, p) > 0.5

    def test_cas_limite_une_seule_valeur_de_c(self, petit_jeu_donnees):
        X, y = petit_jeu_donnees
        pipe = construire_pipeline(["x1", "x2"], ["cat"], seed=0)
        modele, cv = rechercher_meilleur_C(pipe, X, y, [1.0], cv=2, seed=0)
        assert len(cv) == 1
        assert modele.named_steps["clf"].C == 1.0


def test_extraire_coefficients_triee_par_poids_absolu(petit_jeu_donnees):
    X, y = petit_jeu_donnees
    pipe = construire_pipeline(["x1", "x2"], ["cat"], seed=0)
    pipe.fit(X, y)
    table = extraire_coefficients(pipe, ["x1", "x2"], ["cat"])
    assert list(table.columns) == ["variable", "coefficient"]
    poids = table.coefficient.abs().values
    assert all(poids[i] >= poids[i + 1] for i in range(len(poids) - 1))
    assert "x1" in set(table.variable)  # x1 porte le signal (voir conftest)


# ============================ audit d'équité ================================

class TestResumerDecisionEquite:
    def test_nominal(self, capsys):
        df = pd.DataFrame({
            "groupe": ["a", "a", "b", "b"],
            "gt_bon": [0, 0, 0, 0],
        })
        approuve = [1, 1, 0, 1]
        taux, ratio, refus = resumer_decision_equite(df, approuve, "groupe", gt_true_col="gt_bon")
        assert taux["a"] == pytest.approx(1.0)
        assert taux["b"] == pytest.approx(0.5)
        assert ratio == pytest.approx(0.5)
        assert refus["b"] == pytest.approx(0.5)

    def test_cas_limite_aucune_approbation_dans_un_groupe(self):
        # taux.max() > 0 mais un groupe à 0 -> ratio = 0/max = 0.0, pas de division par zéro
        df = pd.DataFrame({"groupe": ["a", "a", "b", "b"]})
        approuve = [0, 0, 1, 1]
        taux, ratio, _ = resumer_decision_equite(df, approuve, "groupe")
        assert taux["a"] == 0.0
        assert ratio == pytest.approx(0.0)

    def test_cas_limite_aucune_approbation_nulle_part(self):
        # taux.max() == 0 -> le code retourne explicitement 0.0 (pas de NaN)
        df = pd.DataFrame({"groupe": ["a", "a", "b", "b"]})
        approuve = [0, 0, 0, 0]
        _, ratio, _ = resumer_decision_equite(df, approuve, "groupe")
        assert ratio == 0.0


def test_audit_equite_utilise_le_seuil_median(capsys):
    df = pd.DataFrame({"groupe": ["a", "a", "b", "b"]})
    p = np.array([0.1, 0.9, 0.2, 0.8])
    taux, ratio, _ = audit_equite(df, p, "groupe")
    # seuil = médiane(p) = 0.5 ; approuve = p < seuil
    assert taux["a"] == pytest.approx(0.5)
    assert taux["b"] == pytest.approx(0.5)


class TestSeuilsEgaliteOpportunite:
    def test_nominal_et_application(self):
        df = pd.DataFrame({
            "groupe": ["a"] * 10 + ["b"] * 10,
            "gt_bon": [0] * 20,
        })
        rng = np.random.default_rng(0)
        p = rng.uniform(size=20)
        seuils = seuils_egalite_opportunite(df, p, "groupe", "gt_bon")
        assert set(seuils.keys()) == {"a", "b"}

        decision = appliquer_seuils_groupe(df, p, "groupe", seuils)
        assert set(np.unique(decision)).issubset({0, 1})
        assert len(decision) == len(df)


# ============================ SHAP ==========================================

def test_calculer_shap_et_sous_groupe(petit_jeu_donnees):
    X, y = petit_jeu_donnees
    pipe = construire_pipeline(["x1", "x2"], ["cat"], seed=0)
    pipe.fit(X, y)

    valeurs, base = calculer_shap(pipe, X, X.iloc[:20], echantillon_fond=50, seed=0)
    assert valeurs.shape[0] == 20
    assert isinstance(base, float)

    meta = X.iloc[:20].assign(_grp=["g1" if i % 2 == 0 else "g2" for i in range(20)])
    table = shap_par_sousgroupe(valeurs, meta, "_grp", top_n=3)
    assert set(table.columns) == {"g1", "g2"}
    assert len(table) <= 3


# ============================ gain financier ================================

class TestDefautsEvitesATaux:
    def test_nominal(self):
        y = np.array([0, 1, 0, 1, 0])
        p = np.array([0.1, 0.2, 0.3, 0.9, 0.8])  # score croissant = plus sûr
        n_a, n_def = defauts_evites_a_taux(y, p, taux_approbation=0.4)
        assert n_a == 2
        assert n_def == y[np.argsort(p)[:2]].sum()

    def test_cas_limite_taux_zero(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.2, 0.3, 0.4])
        n_a, n_def = defauts_evites_a_taux(y, p, taux_approbation=0.0)
        assert n_a == 0
        assert n_def == 0


def test_grille_gain_provisionnement_colonnes_et_signes():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    p_ancien = rng.uniform(size=n)
    p_nouveau = np.clip(p_ancien - 0.1 * y + rng.normal(0, 0.01, n), 0, 1)  # légèrement meilleur

    table = grille_gain_provisionnement(
        y, p_ancien, p_nouveau, grille_taux_approbation=[0.5, 0.8],
        montant_moyen_credit=1_000_000, taux_provisionnement=0.1, volume_annuel=1000,
    )
    assert list(table.taux_approbation) == [0.5, 0.8]
    assert "gain_fcfa_an" in table.columns


def test_bootstrap_gain_provisionnement_intervalle_coherent():
    rng = np.random.default_rng(0)
    n = 150
    y = rng.integers(0, 2, size=n)
    p_ancien = rng.uniform(size=n)
    p_nouveau = np.clip(p_ancien - 0.1 * y, 0, 1)

    moyenne, (lo, hi) = bootstrap_gain_provisionnement(
        y, p_ancien, p_nouveau, taux_approbation=0.7,
        montant_moyen_credit=1_000_000, taux_provisionnement=0.1, volume_annuel=500,
        n_boot=200, seed=0,
    )
    assert lo <= hi


# ============================ repondération (Kamiran & Calders) =============

class TestPoidsReponderation:
    def test_nominal_formule(self):
        # 2 groupes équilibrés, label déséquilibré uniquement dans le groupe "a"
        y = [0, 0, 0, 1, 0, 1, 1, 1]
        g = ["a", "a", "a", "a", "b", "b", "b", "b"]
        poids = poids_reponderation(y, g)
        # P(g=a)=0.5, P(y=0)=0.5, P(g=a,y=0)=3/8 -> poids = 0.25/0.375
        attendu_a0 = (0.5 * 0.5) / (3 / 8)
        idx_a0 = [i for i in range(8) if g[i] == "a" and y[i] == 0]
        assert poids[idx_a0[0]] == pytest.approx(attendu_a0)

    def test_cas_limite_pas_de_desequilibre_poids_uniformes(self):
        y = [0, 1, 0, 1]
        g = ["a", "a", "b", "b"]
        poids = poids_reponderation(y, g)
        assert np.allclose(poids, 1.0)


# ============================ bootstrap AUC ==================================

class TestBootstrapAucCi:
    def test_nominal_intervalle_contient_lestimation_ponctuelle_ou_est_proche(self):
        rng = np.random.default_rng(0)
        n = 300
        y = rng.integers(0, 2, size=n)
        p = np.clip(y * 0.6 + rng.normal(0, 0.3, n), 0, 1)
        auc_obs, (lo, hi) = bootstrap_auc_ci(y, p, n_boot=300, seed=0)
        assert lo <= hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert lo <= auc_obs + 1e-9

    def test_cas_limite_petit_echantillon_ne_leve_pas(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.2, 0.4, 0.6, 0.8])
        auc_obs, (lo, hi) = bootstrap_auc_ci(y, p, n_boot=100, seed=0)
        assert np.isfinite(auc_obs)
        assert np.isfinite(lo) and np.isfinite(hi)
