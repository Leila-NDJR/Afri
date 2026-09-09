#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Catégorisation des libellés de transaction par lexique de règles (palier M2).

Partagé entre m2_m3_texte.ipynb et equite.ipynb, qui ont tous les deux besoin
des mêmes parts de catégories par client.
"""

import re
import unicodedata

import pandas as pd

# une racine courte suffit à identifier une catégorie, même tronquée
LEX = {
    "SALAIRE":   ["SAL", "PAIE", "VIR", "EMPLOY"],
    "ELEC":      ["ENEO", "COURAN", "ELEC", "PREPAID"],
    "EAU":       ["CAMWAT", "EAU", "WATER"],
    "LOYER":     ["LOYER", "BAIL", "MAISON"],
    "MOMO":      ["OM", "MOMO", "MTN", "ORANG", "CASH", "RETRA", "DEPOT", "ENVOI", "MNY"],
    "DASH":      ["DASH", "PRET", "AVANCE", "REMB", "RBST"],
    "NJANGUI":   ["NJANG", "TONT", "COTIS"],
    "CHOP":      ["CHOP", "MARCH", "PROV", "COURS", "ALIMENT"],
    "TRANSPORT": ["TAXI", "MOTO", "BENSK", "CARB", "CLANDO"],
    "PARI":      ["BET", "1XBET", "MELBET", "PMUC", "PARI", "COUPON", "MISE"],
    "AGIOS":     ["AGIO", "DECOUV", "FRAIS", "REJET", "COMM"],
}


def normaliser_libelle(libelle):
    """Nettoie un libellé brut pour résister aux séparateurs mangés et coquilles."""
    s = unicodedata.normalize("NFKD", str(libelle)).encode("ascii", "ignore").decode().upper()
    # codes techniques (référence, terminal, agence) : "AG(?!IO)" protège la racine
    # AGIOS/AGIO (catégorie AGIOS) de la suppression — sans cette exception négative,
    # "AGIOS"/"AGIO" est lui-même reconnu comme un code d'agence (préfixe "AG") et
    # disparaît avant la recherche de racines, ce qui faisait chuter à tort ~40 % des
    # transactions AGIOS réelles en INCONNU (bug détecté par les tests unitaires,
    # voir tests/test_categorisation.py).
    s = re.sub(r"(REF|TPE|AG(?!IO))\w*", " ", s)
    s = re.sub(r"\d", " ", s)               # chiffres (référence, téléphone, date)
    s = re.sub(r"[^A-Z]", " ", s)           # ponctuation / séparateurs
    return re.sub(r"\s+", "", s)


def categoriser_libelle(libelle):
    """Retrouve la catégorie d'un libellé par comptage de racines du lexique."""
    normalise = normaliser_libelle(libelle)
    meilleure_categorie, meilleur_score = "INCONNU", 0
    for categorie, racines in LEX.items():
        score = sum(racine in normalise for racine in racines)
        if score > meilleur_score:
            meilleure_categorie, meilleur_score = categorie, score
    return meilleure_categorie


def construire_parts_categories(tx):
    """Catégorise chaque transaction, puis calcule la part de chaque catégorie par client."""
    tx = tx.copy()
    tx["cat_regle"] = tx.libelle.map(categoriser_libelle)
    parts = tx.groupby(["client_id", "cat_regle"]).size().unstack(fill_value=0)
    parts = parts.div(parts.sum(axis=1), axis=0)
    parts.columns = ["PART_" + c for c in parts.columns]
    return parts.reset_index(), tx
