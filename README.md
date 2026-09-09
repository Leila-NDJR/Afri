# Afri — Scoring de crédit comportemental sous agent LLM local

**Auteur : K. Jessy**
Projet réalisé dans le cadre d'un stage BEAC / du protocole de recherche TOFE-Congo (ISSEA, Master en data science, modélisation statistique).

## Contexte

Selon la Politique de crédit 2025 d'Afriland First Bank, le taux de casse des crédits aux entreprises dépasse 13 %, excédant la norme interne de 6 %, avec un montant d'anomalies atteignant 170 milliards de FCFA au 31 décembre 2024. Ce projet construit et audite un pipeline de **scoring de crédit comportemental** en six paliers (M0 → M6) pour le segment 18-35 ans du marché camerounais (Camfranglais compris), en s'appuyant sur les données déclaratives, comportementales, transactionnelles et textuelles des clients, jusqu'à l'intégration d'un **agent LLM local** (Ollama, `llama3.1`, 100 % hors-ligne) qui attribue trois scores qualitatifs additionnels (S1 : volonté de remboursement, S2 : cohérence niveau de vie / revenu, S3 : risque sectoriel).

Le projet inclut un audit « expert scoring comportemental + IA responsable » qui a identifié et corrigé des failles de reproductibilité et documenté un biais sectoriel non neutralisé dans le pipeline LLM (voir section Résultats clés ci-dessous).

## Structure du projet

```
Afri/
├── Générateur_données.ipynb   # Génération du jeu de données synthétique (clients + transactions)
├── baseline.ipynb             # Palier M0/M1 — modèle déclaratif de référence
├── m2_m3_texte.ipynb          # Paliers M2/M3 — catégorisation des libellés de transaction, modèle comportemental + texte
├── m4_embeddings.ipynb        # Palier M4 — embeddings CamemBERT sur le texte des transactions
├── m5_m6_llm.ipynb            # Paliers M5/M6 — pilote de l'agent LLM local (scores S1/S2/S3), fusion aux modèles statistiques
├── equite.ipynb               # Audit d'équité (règle des 4/5, disparate impact secteur formel/informel) et correction (Kamiran & Calders, 2012)
├── gain_financier.ipynb       # Traduction des performances du modèle en gain financier
├── shap_sousgroupe.ipynb      # Explicabilité SHAP par sous-groupe
├── categorisation.py          # Lexique de règles pour catégoriser les libellés de transaction (palier M2)
├── scoring_utils.py           # Fonctions partagées : pipeline de modélisation, AUC/Gini/KS, test de DeLong
├── llm_utils.py                # Agent LLM local (Ollama) : construction du prompt, appel, validation, scores S1/S2/S3
├── tests/                     # Tests unitaires (pytest) — categorisation, scoring_utils, llm_utils
├── clients_synth.csv          # Jeu de données synthétique : 2 000 clients
├── transactions_synth.csv     # Jeu de données synthétique : 91 666 transactions bancaires
├── Protocole_version_2.docx   # Protocole de recherche (référence académique du projet)
└── requirements.txt           # Dépendances Python
```

Le diaporama de présentation des corrections d'audit (`build_afri.py` → `afri_presentation.pptx`) se trouve dans `../presentations/` (dossier partagé avec d'autres projets, avec sa boîte à outils commune `deck_kit.py`).

## Résultats clés

- **Échelle des données** : 2 000 clients (929 secteur formel / 1 071 informel), 91 666 transactions bancaires (3 derniers mois par client), taux de défaut à 90 jours de 15,9 %.
- **Performance des modèles** (échantillon pilote M5/M6, 30 train / 30 test) :

  | Modèle | AUC | Gini | KS |
  |---|---|---|---|
  | M3 (comportemental + texte) | 0,496 | -0,008 | 0,240 |
  | M5 (M3 + scores LLM) | 0,496 | -0,008 | 0,200 |
  | M4 (embeddings CamemBERT) | 0,672 | 0,344 | 0,440 |
  | M6 (M4 + scores LLM) | 0,688 | 0,376 | 0,440 |

  L'apport des scores LLM (M5 vs M3, M6 vs M4) n'est pas statistiquement significatif au test de DeLong sur cet échantillon réduit (p = 1,00 et p = 0,28 respectivement).

- **Fiabilité test-retest (ICC)** — sur 20 dossiers rejoués du pilote LLM, aucun des trois scores n'atteint le seuil de fiabilité de 0,75 (Koo & Mae, 2016) :
  - S1 (volonté de remboursement) : **0,584**
  - S2 (cohérence niveau de vie / revenu) : **0,318**
  - S3 (risque sectoriel) : **0,393**

  Ces valeurs proviennent de scores produits par l'ancien pipeline (température 0,7 dès le premier appel). Après correction, la température de production par défaut est désormais 0,0 (déterministe) dès le premier appel — la valeur 0,7 reste disponible mais explicite, réservée à l'exploration.

- **Biais sectoriel documenté (non neutralisé)** : le secteur déclaré (formel/informel) reste visible du LLM dans le prompt utilisé pour produire S3. Sur l'échantillon pilote, ratio d'approbation par secteur = 0,77 (disparate impact sous le seuil des 4/5), et refus des « vrais bons » : 38,5 % (secteur formel) contre 61,5 % (secteur informel).

## Recommandations

Sur la base de ces résultats, quatre recommandations sont formulées à l'attention d'un comité scientifique / réglementaire avant tout usage du dispositif en décision réelle :

1. **Ne pas conclure à la fiabilité du dispositif en l'état** — refaire le test-retest sur les 200 dossiers prévus au protocole (contre 60, dont 20 rejoués, dans le pilote) avant toute conclusion de fiabilité ; la puissance statistique actuelle est insuffisante (p = 0,220, non significatif).
2. **Ré-exécuter le pilote LLM en mode production (T=0) avant tout usage en décision** — les ICC mesurés proviennent de scores produits à T=0,7 et doivent être remesurés avec le nouveau défaut déterministe avant d'être opposés à un client.
3. **Sortir S3 (risque sectoriel) de l'agent LLM avant tout usage réel** — le brancher sur le modèle statistique déjà corrigé pour l'équité (M3 repondéré, `equite.ipynb`) plutôt que sur le raisonnement opaque du LLM, qui reproduit un biais sectoriel non neutralisé.
4. **Auditer le biais du LLM sur des données réelles, pas seulement synthétiques, avant déploiement** — les chiffres d'équité et de fiabilité ci-dessus proviennent d'un jeu synthétique et d'un pilote restreint ; un audit sur dossiers réels, à effectif suffisant, est un préalable au déploiement en production.

Le détail de ces recommandations et des corrections déjà apportées à `llm_utils.py` est présenté dans le diaporama `afri_presentation.pptx`.

## Installation

Les dépendances Python sont listées dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

L'agent LLM (`llm_utils.py`) suppose [Ollama](https://ollama.com) installé et démarré localement, avec le modèle `llama3.1` déjà tiré :

```bash
ollama pull llama3.1
```

### Tests

```bash
pytest
```

Les tests unitaires (`tests/`) couvrent notamment la catégorisation des transactions, les fonctions de modélisation partagées (`scoring_utils.py`) et le pipeline de l'agent LLM (`llm_utils.py`), y compris le verrouillage par `monkeypatch` du défaut de température déterministe.
