#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixtures partagées : petit jeu de données synthétique, rapide à entraîner,
avec un signal appris nettement (AUC très supérieure à 0,5) pour que les
assertions sur les métriques soient discriminantes sans être fragiles."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def petit_jeu_donnees():
    """120 lignes, 2 variables numériques + 1 catégorielle, target corrélée à x1."""
    rng = np.random.default_rng(0)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b"], size=n)
    logit = 2.5 * x1 - 0.3 * x2
    proba = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < proba).astype(int)

    X = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})
    return X, y


@pytest.fixture
def petit_jeu_donnees_texte(petit_jeu_donnees):
    """Même jeu de données + une colonne de texte pour tester la branche TF-IDF+SVD."""
    X, y = petit_jeu_donnees
    rng = np.random.default_rng(1)
    mots = ["salaire virement", "achat marche chop", "agios decouvert frais", "pari mise bet"]
    X = X.copy()
    X["texte"] = rng.choice(mots, size=len(X))
    return X, y
