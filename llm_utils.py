#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent LLM local (scores S1/S2/S3) et pipeline de validation formelle.

Écart au protocole documenté dans `m5_m6_llm.ipynb` : modèle `llama3.1`
(Ollama, Q4_K_M) à la place de Mistral 7B / Qwen2.5-7B — seul modèle
disponible localement dans cet environnement.

Reproductibilité (température) — corrigé suite audit.
----------------------------------------------------
Avant correction, la température 0 (déterministe) n'intervenait qu'en
retry, après un échec (JSON invalide ou justification trop courte) : le
tout premier appel — celui qui produit le score utilisé pour la quasi-
totalité des dossiers, puisque c'est le cas le plus fréquent — restait à
température 0.7. Un score de crédit individuel obtenu par un seul appel à
température non nulle n'a par construction aucune garantie de fiabilité
test-retest, ce qui est inacceptable pour une décision réglementaire
traçable.

Correction retenue (option la plus simple, cohérente avec le pipeline de
retry déjà déterministe) : `scorer_un_dossier`/`scorer_dossiers` utilisent
désormais `TEMPERATURE_PRODUCTION = 0.0` par défaut pour TOUS les appels,
y compris le tout premier. `TEMPERATURE_EXPLORATOIRE = 0.7` reste
disponible en passant explicitement `temperature=TEMPERATURE_EXPLORATOIRE`
(utile en exploration/développement — diversité des formulations de
justification — jamais pour un score utilisé en décision).

Note : la température 0 réduit fortement, mais ne garantit pas à 100 %,
l'identité bit à bit entre deux appels (implémentations de sampling,
kernels non déterministes) — c'est pourquoi l'ICC test-retest (voir plus
bas) reste mesuré plutôt que supposé égal à 1.

Biais sectoriel (score S3) — risque documenté, PAS neutralisé.
----------------------------------------------------------
`equite.ipynb` a mesuré et corrigé (seuils différenciés par groupe et
repondération, voir `scoring_utils.seuils_egalite_opportunite` /
`poids_reponderation`) un biais sectoriel (secteur formel/informel) sur
les modèles statistiques M0-M3. Ce module (`construire_dossier_texte`,
`PROMPT_TEMPLATE`) transmet cependant encore le secteur déclaré du client
au LLM pour produire S3 (risque sectoriel) : rien n'empêche l'agent de
reproduire, via un raisonnement opaque, le même type de corrélation
secteur -> risque déjà mise en cause en amont — voir le repère laissé
dans `m5_m6_llm.ipynb` (Étape 8) et le commentaire au point d'injection
ci-dessous.

Pourquoi ce n'est PAS corrigé ici (documenté honnêtement plutôt qu'un
correctif superficiel) : la correction propre supposerait de sortir S3 du
LLM et de le calculer à partir d'une table de risque sectoriel
indépendante et déjà auditée pour l'équité. Or aucune table de ce type
n'existe ailleurs dans le projet (`categorisation.py`, `scoring_utils.py`)
— la seule utilisation de `secteur` y est comme variable catégorielle
d'un modèle déjà corrigé par seuils/repondération, pas comme un score
sectoriel isolé réutilisable. La construire naïvement à partir du taux de
défaut observé (`defaut_90j`) par secteur ré-encoderait très probablement
le même biais d'étiquette déjà identifié et documenté dans `equite.ipynb`
(section « ce biais est en partie construit par le générateur »,
`BIAIS_INFORMEL_FLIP`) — donc un correctif "table de risque sectoriel"
fait à la légère ne ferait que déplacer le problème d'un raisonnement LLM
opaque vers un chiffre en apparence neutre mais tout aussi biaisé.
Masquer `secteur` dans le prompt sans lui substituer un signal de
remplacement rendrait S3 non informatif (le LLM n'aurait alors plus rien
sur quoi fonder ce score). Une vraie neutralisation demanderait donc de
sortir S3 du schéma `ScoreLLM` et de le brancher sur le modèle
statistique déjà corrigé pour l'équité (M3 repondéré ou à seuils par
groupe) — une refonte du pipeline M5/M6 qui dépasse le périmètre de ce
correctif ciblé.
"""

import json
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

MODEL_LLM = "llama3.1"
JUSTIFICATION_MIN = 80
SCORE_MEDIANE = 3

# Mode production (déterministe, par défaut partout dans ce module — voir docstring
# du module, section reproductibilité) vs mode exploratoire (ancien comportement,
# T=0.7, à ne jamais utiliser pour un score effectivement utilisé en décision).
TEMPERATURE_PRODUCTION = 0.0
TEMPERATURE_EXPLORATOIRE = 0.7


# ============================ schéma et prompt =============================

class ScoreLLM(BaseModel):
    """Schéma Pydantic V2 des trois scores qualitatifs et de leur justification."""
    S1: int
    S2: int
    S3: int
    justification: str


# NB biais (voir docstring du module) : S3 ci-dessous est calculé par le LLM à partir
# du `secteur` déclaré injecté par `construire_dossier_texte`, sans neutralisation —
# risque de réintroduire le biais sectoriel déjà corrigé en amont dans `equite.ipynb`.
PROMPT_TEMPLATE = """Tu es un analyste crédit dans une banque camerounaise, spécialisé dans le segment 18-35 ans et familier du Camfranglais (mélange français/anglais/argot). Tu analyses un dossier de demande de crédit à partir des informations déclaratives et du comportement transactionnel observé sur ses 3 derniers mois de relevé de compte.

Dossier client :
{dossier_texte}

Attribue trois scores de 1 (très défavorable) à 5 (très favorable) :
- S1 : volonté de remboursement (régularité des paiements, signes de détresse financière dans les libellés)
- S2 : cohérence entre le niveau de vie apparent (dépenses observées) et le revenu déclaré
- S3 : niveau de risque du secteur d'activité déclaré

Réponds STRICTEMENT en JSON valide, sans texte avant ou après, avec exactement ces clés :
{{"S1": <entier 1-5>, "S2": <entier 1-5>, "S3": <entier 1-5>, "justification": "<80 à 200 caractères expliquant les trois scores>"}}
"""


def construire_dossier_texte(client_row, comp_row, libelles_client, n_libelles=8, seed=0):
    """Résumé lisible d'un dossier (déclaratif + comportemental + échantillon de libellés bruts).

    ATTENTION biais (voir docstring du module, section « Biais sectoriel ») : le
    `secteur` déclaré est transmis tel quel au LLM, qui l'utilise pour produire S3
    (risque sectoriel) — un signal déjà identifié comme porteur de biais et corrigé
    en amont sur les modèles statistiques (`equite.ipynb`), mais PAS neutralisé ici.
    Risque documenté, pas encore corrigé : cf. docstring du module pour le détail
    et pourquoi une vraie neutralisation dépasse le périmètre de ce correctif.
    """
    rng = np.random.default_rng(seed + int(client_row.client_id))
    libelles = list(libelles_client)
    if len(libelles) > n_libelles:
        libelles = list(rng.choice(libelles, size=n_libelles, replace=False))

    return (
        # secteur injecté ici et lu par le LLM pour S3 (risque sectoriel) — voir
        # l'avertissement dans le docstring de cette fonction / du module.
        f"- âge : {client_row.age}, secteur : {client_row.secteur}, zone : {client_row.zone}, "
        f"région : {client_row.region}\n"
        f"- revenu déclaré : {client_row.revenu_declare:.0f} FCFA/mois, "
        f"ancienneté bancaire : {client_row.anciennete_mois} mois\n"
        f"- comportement transactionnel (3 mois) : {comp_row.nb_tx:.0f} transactions, "
        f"{comp_row.pct_debits:.0%} de débits, entrées {comp_row.inflow:.0f} FCFA, "
        f"sorties {comp_row.outflow:.0f} FCFA, {comp_row.nb_jours_actifs:.0f} jours actifs, "
        f"écart revenu déclaré/observé : {comp_row.ecart_revenu:+.0%}\n"
        f"- libellés typiques observés : {' ; '.join(libelles)}"
    )


# ============================ appel au modèle ===============================

def interroger_llm(dossier_texte, model=MODEL_LLM, temperature=TEMPERATURE_PRODUCTION,
                    num_predict=200, num_ctx=1024):
    """Un appel au modèle local, réponse contrainte au format JSON.

    `temperature=TEMPERATURE_PRODUCTION` (0.0) par défaut, cohérent avec le reste du
    module : voir docstring du module (section reproductibilité) pour le raisonnement.
    Passer `temperature=TEMPERATURE_EXPLORATOIRE` explicitement pour l'ancien
    comportement (T=0.7), réservé à l'exploration/développement.
    """
    if ollama is None:
        raise RuntimeError(
            "Le paquet 'ollama' n'est pas installé ou Ollama n'est pas démarré "
            "(pip install ollama, puis 'ollama serve')."
        )
    reponse = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(dossier_texte=dossier_texte)}],
        format="json",
        options={"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    )
    return reponse["message"]["content"]


# ============================ pipeline de validation =========================

def parser_json(reponse_brute):
    """Étape 1 du pipeline : `json.loads` + schéma Pydantic (types). (dict ou None, erreur ou None)."""
    try:
        data = json.loads(reponse_brute)
        score = ScoreLLM(**data)
        return score.model_dump(), None
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        return None, str(e)


def imputer_hors_plage(scores):
    """Étape 2 : impute la médiane (3) pour tout score hors [1, 5], et signale le flag."""
    hors_plage = False
    for cle in ("S1", "S2", "S3"):
        if not (1 <= scores[cle] <= 5):
            scores[cle] = SCORE_MEDIANE
            hors_plage = True
    return scores, hors_plage


def _appeler_llm_sans_lever(dossier_texte, model, temperature):
    """Appelle le LLM ; une panne réseau/Ollama est renvoyée comme une erreur de
    parsing ordinaire plutôt que de remonter et interrompre tout le batch."""
    try:
        return interroger_llm(dossier_texte, model=model, temperature=temperature), None
    except Exception as e:
        return None, f"appel LLM échoué : {e}"


def scorer_un_dossier(dossier_texte, model=MODEL_LLM, temperature=TEMPERATURE_PRODUCTION):
    """Un dossier à travers tout le pipeline (JSON -> plage -> justification), avec un seul retry T=0.

    `temperature=TEMPERATURE_PRODUCTION` (0.0) par défaut : le tout premier appel — pas
    seulement le retry — est déterministe, pour garantir la reproductibilité du score
    utilisé en décision (voir docstring du module, correction suite audit : avant cette
    correction, T=0 n'intervenait qu'en retry après échec, jamais sur le score produit
    par un premier appel réussi, le cas le plus fréquent). Passer explicitement
    `temperature=TEMPERATURE_EXPLORATOIRE` (0.7) pour l'ancien comportement, réservé à
    l'exploration — jamais pour un score utilisé en décision individuelle de crédit.
    Le retry déterministe (après JSON invalide ou justification trop courte) reste à
    T=0 quel que soit `temperature` choisi pour le premier appel.

    Retourne un dict {S1, S2, S3, justification, json_invalide, hors_plage, justification_courte}.
    `json_invalide=True` -> dossier à écarter de M5/M6 (S1/S2/S3 = NaN). Une panne réseau/Ollama
    (broker injoignable, timeout, modèle déchargé) est traitée de la même façon : le dossier est
    écarté plutôt que de faire planter tout `scorer_dossiers`.
    """
    reponse, erreur = _appeler_llm_sans_lever(dossier_texte, model, temperature=temperature)
    data, erreur = (None, erreur) if erreur is not None else parser_json(reponse)

    if erreur is not None:
        # retry déterministe (T=0) avant d'abandonner le dossier
        reponse, erreur = _appeler_llm_sans_lever(dossier_texte, model, temperature=0.0)
        data, erreur = (None, erreur) if erreur is not None else parser_json(reponse)
        if erreur is not None:
            return {"S1": np.nan, "S2": np.nan, "S3": np.nan, "justification": None,
                    "json_invalide": True, "hors_plage": False, "justification_courte": False}

    data, hors_plage = imputer_hors_plage(data)
    justification_courte = len(data["justification"]) < JUSTIFICATION_MIN

    if justification_courte:
        reponse2, erreur2 = _appeler_llm_sans_lever(dossier_texte, model, temperature=0.0)
        data2, erreur2 = (None, erreur2) if erreur2 is not None else parser_json(reponse2)
        if erreur2 is None:
            data2, hors_plage2 = imputer_hors_plage(data2)
            if len(data2["justification"]) >= JUSTIFICATION_MIN:
                data, hors_plage, justification_courte = data2, hors_plage2, False

    return {**data, "json_invalide": False, "hors_plage": hors_plage,
            "justification_courte": justification_courte}


def scorer_dossiers(dossiers, id_col="client_id", texte_col="dossier_texte",
                     model=MODEL_LLM, temperature=TEMPERATURE_PRODUCTION,
                     verbose_every=20, checkpoint_path=None):
    """Score une table de dossiers (une ligne par client) ; affiche la progression.

    `temperature=TEMPERATURE_PRODUCTION` (0.0) par défaut — voir `scorer_un_dossier` et
    la docstring du module. Passer `temperature=TEMPERATURE_EXPLORATOIRE` pour le mode
    exploratoire (jamais pour un batch dont les scores serviront à une décision).

    Si `checkpoint_path` est fourni, la progression est sauvegardée sur disque tous les
    `verbose_every` dossiers (et à la fin) : une interruption (Ctrl+C, panne Ollama
    prolongée, crash) ne fait donc perdre au plus que les derniers dossiers en cours,
    pas l'intégralité du batch (potentiellement plusieurs heures sur des centaines de
    dossiers).
    """
    cols = [id_col, "S1", "S2", "S3", "justification", "json_invalide", "hors_plage", "justification_courte"]
    t0 = time.time()
    lignes = []
    for i, row in enumerate(dossiers.itertuples(index=False)):
        res = scorer_un_dossier(getattr(row, texte_col), model=model, temperature=temperature)
        res[id_col] = getattr(row, id_col)
        lignes.append(res)
        if verbose_every and (i + 1) % verbose_every == 0:
            ecoule = time.time() - t0
            print(f"{i + 1}/{len(dossiers)} dossiers scorés ({ecoule / (i + 1):.1f} s/dossier en moyenne)")
            if checkpoint_path:
                pd.DataFrame(lignes)[cols].to_parquet(checkpoint_path, index=False)
    table = pd.DataFrame(lignes)[cols]
    if checkpoint_path:
        table.to_parquet(checkpoint_path, index=False)
    return table


def resumer_validation(scores):
    """Taux de succès du pipeline de validation, pour audit."""
    n = len(scores)
    print(f"dossiers écartés (JSON invalide malgré retry) : {scores.json_invalide.sum()}/{n} "
          f"({scores.json_invalide.mean():.1%})")
    valides = scores[~scores.json_invalide]
    print(f"scores imputés à la médiane (hors plage) : {valides.hors_plage.sum()}/{len(valides)} "
          f"({valides.hors_plage.mean():.1%})")
    print(f"justification jugée trop courte malgré retry : {valides.justification_courte.sum()}/{len(valides)} "
          f"({valides.justification_courte.mean():.1%})")


# ============================ fiabilité test-retest (ICC) ====================

def calculer_icc(scores_p1, scores_p2, score_col, id_col="client_id"):
    """ICC (two-way mixed, consistency = ICC(C,1) dans pingouin) entre deux passes du même dossier.

    À calculer sur les dossiers communs aux deux passes (voir Koo & Mae, 2016) :
    < 0,75 -> score abandonné ; [0,75 ; 0,90[ -> acceptable ; >= 0,90 -> excellent.

    Avertissement échelle (voir `m5_m6_llm.ipynb`) : le protocole prévoit 200 dossiers
    de test-retest (dont un nombre substantiel de défauts) ; le pilote de ce projet n'en
    couvre que 60 (environ 9-10 défauts). Un ICC calculé sur un tel effectif doit être
    présenté comme strictement indicatif (démonstration du pipeline), pas comme une
    validation statistique de la fiabilité test-retest — ne pas sur-interpréter un ICC
    élevé ou faible obtenu à cette échelle comme une conclusion définitive.
    """
    import pingouin as pg

    long = pd.concat([
        scores_p1[[id_col, score_col]].assign(passe="1"),
        scores_p2[[id_col, score_col]].assign(passe="2"),
    ], ignore_index=True)
    icc = pg.intraclass_corr(data=long, targets=id_col, raters="passe", ratings=score_col)
    return icc.set_index("Type").loc["ICC(C,1)", "ICC"]
