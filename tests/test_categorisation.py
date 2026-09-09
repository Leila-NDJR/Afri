#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests unitaires de categorisation.py (lexique de règles, palier M2).

Cas nominaux (libellés propres et bruités connus du lexique) + cas limites
(chaîne vide, valeurs manquantes, libellé hors lexique).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from categorisation import (
    LEX,
    categoriser_libelle,
    construire_parts_categories,
    normaliser_libelle,
)


# ============================ normaliser_libelle ==========================

class TestNormaliserLibelle:
    def test_nominal_accents_chiffres_ponctuation(self):
        # accents retirés, passage en majuscules, chiffres/ponctuation supprimés
        assert normaliser_libelle("paiMt-FAct.EnEo/REF9731984") == "PAIMTFACTENEO"

    def test_nominal_retire_codes_reference_et_terminal(self):
        # les codes REF/TPE/AG sont retirés avant les chiffres/ponctuation isolés
        resultat = normaliser_libelle("DASH PRET REF383522 TPE50248")
        assert "REF" not in resultat
        assert "TPE" not in resultat
        assert "DASHPRET" in resultat

    def test_chaine_vide(self):
        assert normaliser_libelle("") == ""

    def test_valeur_manquante_nan(self):
        # str(np.nan) == "nan" -> normalisé en "NAN", ne doit pas lever d'exception
        resultat = normaliser_libelle(np.nan)
        assert resultat == "NAN"

    def test_valeur_manquante_none(self):
        resultat = normaliser_libelle(None)
        assert resultat == "NONE"


# ============================ categoriser_libelle =========================

class TestCategoriserLibelle:
    @pytest.mark.parametrize("categorie", list(LEX.keys()))
    def test_nominal_chaque_categorie_du_lexique_est_retrouvee(self, categorie):
        # chaque catégorie doit être retrouvée à partir d'une de ses propres racines
        racine = LEX[categorie][0]
        assert categoriser_libelle(f"OPERATION {racine} DIVERSE") == categorie

    def test_nominal_libelle_bruite_reel(self):
        assert categoriser_libelle("paiMt-FAct.EnEo/REF9731984") == "ELEC"

    def test_chaine_vide_est_inconnu(self):
        assert categoriser_libelle("") == "INCONNU"

    def test_libelle_hors_lexique_est_inconnu(self):
        assert categoriser_libelle("ACHAT DIVERS SANS RAPPORT") == "INCONNU"

    def test_valeur_manquante_ne_leve_pas(self):
        assert categoriser_libelle(np.nan) == "INCONNU"

    def test_nominal_plusieurs_racines_de_la_meme_categorie_gagnent(self):
        # deux racines NJANGUI présentes -> score 2, doit dominer une seule racine d'une autre catégorie
        assert categoriser_libelle("NJANG TONT COTIS PRET") == "NJANGUI"


# ============================ construire_parts_categories =================

class TestConstruirePartsCategories:
    def _tx(self, lignes):
        return pd.DataFrame(lignes)

    def test_nominal_parts_somment_a_un_par_client(self):
        tx = self._tx([
            {"client_id": 1, "libelle": "VIR SALAIRE EMPLOYEUR"},
            {"client_id": 1, "libelle": "FACTURE ENEO"},
            {"client_id": 1, "libelle": "FACTURE ENEO"},
            {"client_id": 2, "libelle": "PARI 1XBET MISE"},
        ])
        parts, tx_categorise = construire_parts_categories(tx)

        part_cols = [c for c in parts.columns if c.startswith("PART_")]
        sommes = parts.set_index("client_id")[part_cols].sum(axis=1)
        assert np.allclose(sommes.values, 1.0)
        assert "cat_regle" in tx_categorise.columns
        assert set(parts.client_id) == {1, 2}

    def test_nominal_repartition_correcte(self):
        tx = self._tx([
            {"client_id": 1, "libelle": "FACTURE ENEO"},
            {"client_id": 1, "libelle": "FACTURE ENEO"},
            {"client_id": 1, "libelle": "PARI 1XBET MISE"},
        ])
        parts, _ = construire_parts_categories(tx)
        ligne = parts.set_index("client_id").loc[1]
        assert ligne["PART_ELEC"] == pytest.approx(2 / 3)
        assert ligne["PART_PARI"] == pytest.approx(1 / 3)

    def test_cas_limite_un_seul_client_une_seule_transaction(self):
        tx = self._tx([{"client_id": 7, "libelle": "ACHAT DIVERS SANS RAPPORT"}])
        parts, _ = construire_parts_categories(tx)
        assert list(parts.client_id) == [7]
        assert parts.set_index("client_id").loc[7, "PART_INCONNU"] == 1.0

    def test_cas_limite_libelle_manquant(self):
        tx = self._tx([
            {"client_id": 1, "libelle": np.nan},
            {"client_id": 1, "libelle": "FACTURE ENEO"},
        ])
        # ne doit pas lever d'exception malgré la valeur manquante
        parts, tx_categorise = construire_parts_categories(tx)
        assert parts.set_index("client_id").loc[1].sum() == pytest.approx(1.0)
