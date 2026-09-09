#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests unitaires de llm_utils.py (pipeline de validation formelle de l'agent LLM).

`ollama.chat` est mocké partout : ces tests ne nécessitent ni Ollama installé,
ni service local démarré. Ils couvrent le cœur des hypothèses H3/H4 du
protocole : JSON invalide, scores hors plage, justification trop courte,
et — ajouté ici — une panne réseau/Ollama qui ne doit pas faire planter le
pipeline (cf. rapport d'analyse : ce cas n'avait aucune gestion d'erreur).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_utils
from llm_utils import (
    JUSTIFICATION_MIN,
    SCORE_MEDIANE,
    TEMPERATURE_EXPLORATOIRE,
    TEMPERATURE_PRODUCTION,
    imputer_hors_plage,
    parser_json,
    scorer_dossiers,
    scorer_un_dossier,
)


def _json_scores(s1=4, s2=3, s3=2, justification=None):
    if justification is None:
        justification = "x" * JUSTIFICATION_MIN  # juste assez longue
    return f'{{"S1": {s1}, "S2": {s2}, "S3": {s3}, "justification": "{justification}"}}'


# ============================ parser_json ==================================

class TestParserJson:
    def test_json_valide(self):
        data, erreur = parser_json(_json_scores(4, 3, 2))
        assert erreur is None
        assert data == {"S1": 4, "S2": 3, "S3": 2, "justification": "x" * JUSTIFICATION_MIN}

    def test_json_invalide_syntaxe(self):
        data, erreur = parser_json("{ceci n'est pas du JSON")
        assert data is None
        assert erreur is not None

    def test_json_valide_mais_cle_manquante(self):
        data, erreur = parser_json('{"S1": 4, "S2": 3}')  # S3 et justification absents
        assert data is None
        assert erreur is not None

    def test_json_valide_mais_type_incorrect(self):
        data, erreur = parser_json('{"S1": "haut", "S2": 3, "S3": 2, "justification": "abc"}')
        assert data is None
        assert erreur is not None


# ============================ imputer_hors_plage ============================

class TestImputerHorsPlage:
    def test_scores_dans_la_plage_non_modifies(self):
        scores = {"S1": 1, "S2": 5, "S3": 3}
        resultat, hors_plage = imputer_hors_plage(dict(scores))
        assert resultat == scores
        assert hors_plage is False

    def test_score_hors_plage_impute_a_la_mediane(self):
        scores = {"S1": 7, "S2": 0, "S3": 3}
        resultat, hors_plage = imputer_hors_plage(scores)
        assert resultat["S1"] == SCORE_MEDIANE
        assert resultat["S2"] == SCORE_MEDIANE
        assert resultat["S3"] == 3
        assert hors_plage is True


# ============================ scorer_un_dossier (ollama mocké) ==============

class TestScorerUnDossier:
    def test_nominal(self, monkeypatch):
        monkeypatch.setattr(llm_utils, "interroger_llm", lambda *a, **k: _json_scores(4, 3, 2))
        res = scorer_un_dossier("dossier fictif")
        assert res["json_invalide"] is False
        assert res["hors_plage"] is False
        assert res["justification_courte"] is False
        assert res["S1"] == 4

    def test_json_invalide_sur_les_deux_tentatives_ecarte_le_dossier(self, monkeypatch):
        monkeypatch.setattr(llm_utils, "interroger_llm", lambda *a, **k: "pas du json")
        res = scorer_un_dossier("dossier fictif")
        assert res["json_invalide"] is True
        assert np.isnan(res["S1"]) and np.isnan(res["S2"]) and np.isnan(res["S3"])

    def test_json_invalide_puis_valide_au_retry_recupere_le_dossier(self, monkeypatch):
        appels = {"n": 0}

        def fake_interroger(dossier_texte, model, temperature):
            appels["n"] += 1
            if appels["n"] == 1:
                return "reponse cassee"
            return _json_scores(2, 2, 2)

        monkeypatch.setattr(llm_utils, "interroger_llm", fake_interroger)
        res = scorer_un_dossier("dossier fictif")
        assert res["json_invalide"] is False
        assert res["S1"] == 2

    def test_panne_reseau_ollama_ecarte_le_dossier_sans_lever(self, monkeypatch):
        """Avant correction : une exception réseau/Ollama remontait telle quelle et
        interrompait tout `scorer_dossiers`. Désormais elle est traitée comme un
        JSON invalide -> le dossier est écarté, le pipeline continue."""
        def fake_interroger(*a, **k):
            raise ConnectionError("impossible de joindre le broker Ollama")

        monkeypatch.setattr(llm_utils, "interroger_llm", fake_interroger)
        res = scorer_un_dossier("dossier fictif")  # ne doit pas lever
        assert res["json_invalide"] is True

    def test_justification_trop_courte_declenche_un_retry(self, monkeypatch):
        appels = {"n": 0}

        def fake_interroger(dossier_texte, model, temperature):
            appels["n"] += 1
            if appels["n"] == 1:
                return _json_scores(3, 3, 3, justification="trop court")
            return _json_scores(3, 3, 3, justification="x" * JUSTIFICATION_MIN)

        monkeypatch.setattr(llm_utils, "interroger_llm", fake_interroger)
        res = scorer_un_dossier("dossier fictif")
        assert res["justification_courte"] is False
        assert appels["n"] == 2


# ============================ température (reproductibilité, critique audit) ===

class TestTemperatureProduction:
    """Avant correction : T=0 n'intervenait qu'en retry, jamais sur le premier appel
    (le cas le plus fréquent, quand le premier appel réussit du premier coup). Ces
    tests verrouillent le nouveau comportement : le mode production (par défaut)
    est déterministe dès le premier appel."""

    def test_scorer_un_dossier_appelle_interroger_llm_a_temperature_zero_par_defaut(self, monkeypatch):
        appels = []

        def fake_interroger(dossier_texte, model, temperature):
            appels.append(temperature)
            return _json_scores(4, 3, 2)

        monkeypatch.setattr(llm_utils, "interroger_llm", fake_interroger)
        scorer_un_dossier("dossier fictif")

        assert appels == [0.0]
        assert TEMPERATURE_PRODUCTION == 0.0

    def test_scorer_un_dossier_mode_exploratoire_explicite_utilise_0_7_au_premier_appel(self, monkeypatch):
        appels = []

        def fake_interroger(dossier_texte, model, temperature):
            appels.append(temperature)
            return _json_scores(4, 3, 2)

        monkeypatch.setattr(llm_utils, "interroger_llm", fake_interroger)
        scorer_un_dossier("dossier fictif", temperature=TEMPERATURE_EXPLORATOIRE)

        assert appels == [TEMPERATURE_EXPLORATOIRE]

    def test_scorer_un_dossier_mode_exploratoire_le_retry_reste_deterministe(self, monkeypatch):
        """Même en mode exploratoire (premier appel T=0.7), le retry après JSON
        invalide reste à T=0 (comportement de retry inchangé par cette correction)."""
        appels = []

        def fake_interroger(dossier_texte, model, temperature):
            appels.append(temperature)
            if len(appels) == 1:
                return "pas du json"
            return _json_scores(4, 3, 2)

        monkeypatch.setattr(llm_utils, "interroger_llm", fake_interroger)
        res = scorer_un_dossier("dossier fictif", temperature=TEMPERATURE_EXPLORATOIRE)

        assert appels == [TEMPERATURE_EXPLORATOIRE, 0.0]
        assert res["json_invalide"] is False

    def test_scorer_dossiers_transmet_la_temperature_production_par_defaut(self, monkeypatch):
        appels = []

        def fake_interroger(dossier_texte, model, temperature):
            appels.append(temperature)
            return _json_scores(4, 3, 2)

        monkeypatch.setattr(llm_utils, "interroger_llm", fake_interroger)
        dossiers = pd.DataFrame({"client_id": [1, 2], "dossier_texte": ["a", "b"]})
        scorer_dossiers(dossiers, verbose_every=0)

        assert appels == [0.0, 0.0]


# ============================ interroger_llm : ollama absent ================

def test_interroger_llm_leve_une_erreur_claire_si_ollama_absent(monkeypatch):
    monkeypatch.setattr(llm_utils, "ollama", None)
    with pytest.raises(RuntimeError, match="ollama"):
        llm_utils.interroger_llm("dossier fictif")
