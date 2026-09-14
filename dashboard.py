#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dashboard Streamlit — scoring de crédit comportemental M0-M6 (projet Afri).

Auteur du projet : K. Jessy — stage BEAC / protocole de recherche TOFE-Congo
(ISSEA, Master en data science).

Multi-pages (navigation via st.sidebar) :
  - Vue d'ensemble       : contexte, KPIs clés
  - Scorer un client      : score statistique en direct (pipeline M3 réel,
                             ré-entraîné via scoring_utils.py sur les vraies
                             données), exemples déjà calculés du pilote LLM
  - Équité & fiabilité    : audit d'équité réel (equite.ipynb) et ICC réels
                             (m5_m6_llm.ipynb)
  - Performance des modèles : comparaison M0 à M6 (résultats réels des
                             notebooks / README.md)

Lancement :
    streamlit run dashboard.py

Toutes les métriques affichées proviennent de données ou de résultats déjà
produits par le projet (clients_synth.csv, transactions_synth.csv,
scoring_utils.py, ou les sorties déjà calculées de equite.ipynb /
m5_m6_llm.ipynb / README.md) — aucun chiffre n'est inventé.

Important : le scoring LLM (llm_utils.py) n'est PAS relancé en direct ici.
Ollama n'est pas supposé démarré dans cet environnement de démo ; seuls des
résultats déjà calculés dans m5_m6_llm.ipynb sont montrés (pilote de 60
dossiers, test-retest ICC sur 20 dossiers).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from categorisation import construire_parts_categories
from scoring_utils import construire_pipeline, construire_variables_comportementales

# ==============================================================================
# 1. PALETTE PASTEL & THÈME
# ==============================================================================
PASTEL = {
    "violet": "#C4B5FD",       # violet clair — couleur d'accent principale
    "violet_deep": "#8B7CE0",  # violet un peu plus soutenu (texte sur fond clair)
    "violet_soft": "#F3F0FF",  # fond très clair
    "green": "#A8E6CF",        # succès / low risk
    "orange": "#FFD3A5",       # attention / medium risk
    "pink": "#FFAAA5",         # danger / high risk
    "blue": "#A7C7E7",         # secondaire
    "ink": "#000000",          # texte : noir
    "muted": "#8B87A0",
}
PLOTLY_PASTEL_SEQUENCE = [PASTEL["violet"], PASTEL["blue"], PASTEL["green"], PASTEL["orange"], PASTEL["pink"]]

st.set_page_config(
    page_title="Afri — Scoring comportemental M0-M6",
    layout="wide",
    page_icon="💳",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FDFCFF; }}
    /* Barre d'outils Streamlit par défaut : recolorée pour se fondre dans la
    page, SANS rien masquer (visibility/display) — un précédent essai avait
    caché par erreur le bouton pour rouvrir la barre latérale une fois
    réduite, qui vit dans ce même conteneur. */
    header[data-testid="stHeader"] {{ background-color: #FDFCFF; }}
    section[data-testid="stSidebar"] {{ background-color: {PASTEL["violet_soft"]}; }}
    /* Texte en noir, restreint aux éléments qui portent réellement du texte
    (pas de sélecteur `div` générique : ça avait aussi noirci des icônes -
    ex. le bouton de la barre latérale - rendues via `currentColor`, les
    rendant invisibles sur leur propre fond). Le logo JL reste blanc
    (règle .jl-logo plus bas, après celle-ci donc prioritaire à égalité de
    spécificité). */
    h1, h2, h3, h4, h5, h6, p, li, label, td, th,
    .stMarkdown, [data-testid="stCaptionContainer"],
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: #000000 !important;
    }}
    div[data-testid="stMetric"] {{
        background-color: {PASTEL["violet_soft"]};
        border: 1px solid {PASTEL["violet"]};
        border-radius: 12px;
        padding: 12px 16px;
    }}
    .jl-logo, .jl-logo * {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 56px; height: 56px; border-radius: 14px;
        background: {PASTEL["violet"]};
        font-family: Georgia, 'Times New Roman', serif;
        font-style: italic; font-weight: 700; font-size: 24px;
        color: #FFFFFF !important; letter-spacing: 1px;
        box-shadow: 0 2px 8px rgba(139, 124, 224, 0.35);
    }}
    .jl-header {{ display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }}
    .jl-header-title {{ font-size: 15px; font-weight: 700; }}
    .jl-header-sub {{ font-size: 12px; }}
    .jl-card {{
        background: {PASTEL["violet_soft"]}; border-radius: 12px;
        padding: 16px 20px; border-left: 4px solid {PASTEL["violet"]};
        margin-bottom: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def jl_logo_header():
    st.sidebar.markdown(
        f"""
        <div class="jl-header">
            <div class="jl-logo">JL</div>
            <div>
                <div class="jl-header-title">K. Jessy</div>
                <div class="jl-header-sub">Scoring comportemental M0-M6</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(html: str):
    st.markdown(f'<div class="jl-card">{html}</div>', unsafe_allow_html=True)


def _style_fig(fig, **layout_kwargs):
    """Applique le fond transparent + le texte en noir communs à tous les graphiques."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PASTEL["ink"]),
        **layout_kwargs,
    )
    return fig

# ============================================================================
# Constantes reprises telles quelles des notebooks (mêmes colonnes/paramètres
# que baseline.ipynb / m2_m3_texte.ipynb / equite.ipynb) — pour rejouer
# fidèlement le pipeline M3 en direct sur les vraies données du projet.
# ============================================================================

SEED = 42
TARGET = "defaut_90j"
DECLARATIF_NUM = ["age", "revenu_declare", "anciennete_mois"]
DECLARATIF_CAT = ["zone", "secteur", "region"]
COMPORTEMENTAL_NUM = [
    "nb_tx", "pct_debits", "inflow", "outflow", "net_flow",
    "mean_abs", "std_abs", "max_abs", "cv_abs",
    "nb_jours_actifs", "tx_par_jour", "ecart_revenu",
]
# meilleur C retrouvé par validation croisée pour M3 dans m2_m3_texte.ipynb (cellule 27)
C_M3 = 1

SEUIL_FIABILITE_ICC = 0.75  # Koo & Mae, 2016
SEUIL_REGLE_4_5 = 0.80

# ICC test-retest réels (m5_m6_llm.ipynb, Étape 4 — pilote 20 dossiers rejoués,
# ancien pipeline à température 0,7 dès le premier appel, avant correction du
# défaut de température en production ; voir README.md et llm_utils.py)
ICC_LLM = {
    "S1 — volonté de remboursement": 0.584,
    "S2 — cohérence niveau de vie / revenu": 0.318,
    "S3 — risque sectoriel": 0.393,
}

# Performance pleine échelle (train 1600 / test 400) — résultats réels
# affichés par baseline.ipynb, m2_m3_texte.ipynb et m4_embeddings.ipynb
PERF_PLEINE_ECHELLE = pd.DataFrame([
    {"modele": "M0 (déclaratif)",              "auc": 0.653, "gini": 0.307, "ks": 0.268},
    {"modele": "M1 (+ comportemental)",        "auc": 0.666, "gini": 0.333, "ks": 0.275},
    {"modele": "M2 (+ catégories texte)",      "auc": 0.719, "gini": 0.438, "ks": 0.350},
    {"modele": "M2-XGB",                       "auc": 0.722, "gini": 0.445, "ks": 0.367},
    {"modele": "M2-RF",                        "auc": 0.700, "gini": 0.399, "ks": 0.370},
    {"modele": "M2-SMOTE",                     "auc": 0.716, "gini": 0.431, "ks": 0.348},
    {"modele": "M3 (+ TF-IDF texte libre)",    "auc": 0.718, "gini": 0.437, "ks": 0.327},
    {"modele": "M4 (embeddings CamemBERT)",    "auc": 0.681, "gini": 0.361, "ks": 0.297},
])

# Performance du pilote LLM (60 dossiers, 30 train / 30 test) — README.md /
# m5_m6_llm.ipynb, cellule 21
PERF_PILOTE_LLM = pd.DataFrame([
    {"modele": "M3 (pilote)",           "auc": 0.496,  "gini": -0.008, "ks": 0.240},
    {"modele": "M5 = M3 + scores LLM",  "auc": 0.496,  "gini": -0.008, "ks": 0.200},
    {"modele": "M4 (pilote)",           "auc": 0.672,  "gini": 0.344,  "ks": 0.440},
    {"modele": "M6 = M4 + scores LLM",  "auc": 0.688,  "gini": 0.376,  "ks": 0.440},
])

# Audit d'équité réel — equite.ipynb (secteur formel/informel, modèle M3)
EQUITE_SECTEUR = pd.DataFrame([
    {"variante": "M3 référence",                 "auc_test": 0.719, "ratio_4_5": 0.353,
     "refus_vrais_bons_formel": 0.183, "refus_vrais_bons_informel": 0.704},
    {"variante": "M3 + seuils par groupe (A)",    "auc_test": 0.719, "ratio_4_5": 0.879,
     "refus_vrais_bons_formel": 0.183, "refus_vrais_bons_informel": 0.290},
    {"variante": "M3 + repondération Kamiran & Calders (B)", "auc_test": 0.699, "ratio_4_5": 0.763,
     "refus_vrais_bons_formel": 0.384, "refus_vrais_bons_informel": 0.543},
])

# Audit d'équité réel — equite.ipynb (sexe, modèle M3)
EQUITE_SEXE = pd.DataFrame([
    {"variante": "référence",                  "ratio_4_5": 0.93},
    {"variante": "+ seuils par groupe (A)",     "ratio_4_5": 0.94},
    {"variante": "+ repondération (B)",         "ratio_4_5": 0.90},
])

# Biais sectoriel du score LLM S3 — m5_m6_llm.ipynb, cellule 19 (audit sur M6, pilote)
BIAIS_S3_LLM = {
    "ratio_4_5_secteur": 0.77,
    "refus_vrais_bons_formel": 0.385,
    "refus_vrais_bons_informel": 0.615,
}

# Exemple réel de dossier transmis à l'agent LLM (m5_m6_llm.ipynb, cellule 5,
# premier dossier du pilote) — illustratif, PAS le client sélectionné ci-dessous
EXEMPLE_DOSSIER_LLM = (
    "- âge : 35, secteur : formel, zone : urbaine, région : Adamaoua\n"
    "- revenu déclaré : 160 000 FCFA/mois, ancienneté bancaire : 49 mois\n"
    "- comportement transactionnel (3 mois) : 44 transactions, 82 % de débits, "
    "entrées 3 369 891 FCFA, sorties 7 381 116 FCFA, 36 jours actifs, "
    "écart revenu déclaré/observé : -602 %\n"
    "- libellés typiques observés : LoyeR mENsuEl AGEBW 14/0 ; Pmt ElectriCiTE*T ; "
    "CHop/MARche ; vIR_paiEmEnTlOYER*BaILLeURREF4010655AGBAF ; OpR_pmT ; tRF*ChoP ; "
    "ENCasxt ENVOi AGBER 02/05 ; paImT maisoN REF48799"
)


# ============================================================================
# Chargement des données et entraînement du modèle (mis en cache)
# ============================================================================

@st.cache_data(show_spinner="Chargement des données synthétiques...")
def charger_clients():
    return pd.read_csv("clients_synth.csv")


@st.cache_data(show_spinner="Chargement des transactions synthétiques...")
def charger_transactions():
    return pd.read_csv("transactions_synth.csv")


@st.cache_resource(show_spinner="Entraînement du modèle statistique M3 (comme dans m2_m3_texte.ipynb)...")
def entrainer_m3():
    """Rejoue exactement le pipeline M3 de m2_m3_texte.ipynb sur les vraies données.

    Retourne le pipeline entraîné, la table complète (une ligne par client,
    toutes les variables déjà construites) et l'AUC de test — recalculée en
    direct, donc potentiellement identique ou très proche (mêmes seed/split)
    du 0,718-0,719 déjà rapporté dans le README et les notebooks.
    """
    clients = charger_clients()
    tx = charger_transactions()

    comp = construire_variables_comportementales(tx, clients)
    parts, _ = construire_parts_categories(tx)
    part_cols = [c for c in parts.columns if c.startswith("PART_")]
    texte_par_client = (
        tx.groupby("client_id").libelle.apply(lambda s: " ".join(map(str, s)))
        .rename("texte").reset_index()
    )

    df = (
        clients.merge(comp, on="client_id", how="left")
        .merge(parts, on="client_id", how="left")
        .merge(texte_par_client, on="client_id", how="left")
    )
    df[COMPORTEMENTAL_NUM + part_cols] = df[COMPORTEMENTAL_NUM + part_cols].fillna(0)
    df["texte"] = df["texte"].fillna("")

    y = df[TARGET].values
    colonnes_gt = [c for c in df.columns if c.startswith("gt_")]
    X = df.drop(columns=colonnes_gt + [TARGET, "client_id"])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    num_m2 = DECLARATIF_NUM + COMPORTEMENTAL_NUM + part_cols
    pipeline_m3 = construire_pipeline(num_m2, DECLARATIF_CAT, texte_col="texte", seed=SEED)
    pipeline_m3.set_params(clf__C=C_M3)
    pipeline_m3.fit(Xtr, ytr)

    p_test = pipeline_m3.predict_proba(Xte)[:, 1]
    auc_test = float(roc_auc_score(yte, p_test))

    return pipeline_m3, df, colonnes_gt, auc_test


# ============================================================================
# Pages
# ============================================================================

def page_vue_ensemble():
    st.title("💳 Afri — Scoring de crédit comportemental (M0 → M6)")
    st.caption("Projet réalisé par K. Jessy — stage BEAC / protocole de recherche TOFE-Congo (ISSEA)")

    st.markdown(
        """
À fin 2024, les créances en souffrance atteignaient **16,2 %** de l'encours
de crédit dans la zone CEMAC, soit **2 024 milliards de FCFA** (source :
Commission bancaire de l'Afrique centrale — COBAC). Ce projet construit et
audite un pipeline de **scoring de crédit comportemental en six paliers
(M0 → M6)** pour le segment 18-35 ans du marché camerounais, jusqu'à
l'intégration d'un **agent LLM local** (Ollama, `llama3.1`, 100 %
hors-ligne) qui attribue trois scores qualitatifs additionnels (S1, S2, S3).
        """
    )

    clients = charger_clients()
    tx = charger_transactions()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients synthétiques", f"{len(clients):,}".replace(",", " "))
    col2.metric("Transactions synthétiques", f"{len(tx):,}".replace(",", " "))
    col3.metric("Taux de défaut à 90 jours", f"{clients[TARGET].mean():.1%}")
    meilleur = PERF_PLEINE_ECHELLE.loc[PERF_PLEINE_ECHELLE.auc.idxmax()]
    col4.metric(f"Meilleur modèle statistique ({meilleur['modele']})", f"AUC {meilleur['auc']:.3f}")

    st.subheader("Fiabilité de l'agent LLM (ICC test-retest)")
    c1, c2, c3 = st.columns(3)
    for c, (nom, valeur) in zip((c1, c2, c3), ICC_LLM.items()):
        c.metric(nom, f"{valeur:.3f}", delta=f"{valeur - SEUIL_FIABILITE_ICC:+.3f} vs seuil 0,75",
                  delta_color="inverse")
    st.caption(
        "Aucun des trois scores qualitatifs (S1/S2/S3) n'atteint le seuil de fiabilité de 0,75 "
        "(Koo & Mae, 2016) sur le pilote de 20 dossiers rejoués — voir la page « Équité & fiabilité »."
    )

    st.subheader("Répartition du portefeuille synthétique")
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.pie(
            clients, names="secteur", title="Secteur d'activité déclaré", hole=0.45,
            color_discrete_sequence=PLOTLY_PASTEL_SEQUENCE,
        )
        _style_fig(fig, legend=dict(orientation="h", yanchor="bottom", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        fig = px.histogram(
            clients, x="age", nbins=18, title="Distribution de l'âge (18-35 ans)",
            color_discrete_sequence=[PASTEL["violet"]],
        )
        _style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Structure du projet : `Générateur_données.ipynb` → `baseline.ipynb` (M0/M1) → "
        "`m2_m3_texte.ipynb` (M2/M3) → `m4_embeddings.ipynb` (M4) → `m5_m6_llm.ipynb` (M5/M6, "
        "agent LLM) → `equite.ipynb` (audit d'équité) → `shap_sousgroupe.ipynb` (explicabilité) → "
        "`gain_financier.ipynb` (traduction financière). Voir `README.md` pour le détail."
    )


def page_scorer_client():
    st.title("🔎 Scorer un client")
    st.caption(
        "Score statistique calculé en direct par le pipeline **M3** réel du projet "
        "(`scoring_utils.construire_pipeline`, mêmes variables et même C=1 que `m2_m3_texte.ipynb`), "
        "ré-entraîné sur les vraies données du projet."
    )

    clients = charger_clients()
    pipeline_m3, df, colonnes_gt, auc_test = entrainer_m3()
    st.caption(f"AUC de test recalculée en direct sur ce pipeline : **{auc_test:.3f}** "
               f"(train 1 600 / test 400, même split que les notebooks — référence notebook : 0,718-0,719).")

    mode = st.radio("Sélection du client", ["Choisir un client existant", "Saisie manuelle"], horizontal=True)

    if mode == "Choisir un client existant":
        client_id = st.selectbox("Identifiant client", options=clients.client_id.tolist())
        ligne = df[df.client_id == client_id].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Âge", int(ligne.age))
        c2.metric("Secteur", ligne.secteur)
        c3.metric("Zone", ligne.zone)
        c4.metric("Ancienneté bancaire", f"{int(ligne.anciennete_mois)} mois")
        c5, c6 = st.columns(2)
        c5.metric("Revenu déclaré", f"{ligne.revenu_declare:,.0f} FCFA/mois".replace(",", " "))
        c6.metric("Région", ligne.region)

        X_client = df[df.client_id == client_id].drop(columns=colonnes_gt + [TARGET, "client_id"])
        proba = float(pipeline_m3.predict_proba(X_client)[:, 1][0])
        label_reel = int(ligne[TARGET])

        st.subheader("Score statistique (modèle M3)")
        g1, g2 = st.columns([1, 1])
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": " %"},
                title={"text": "Probabilité de défaut à 90 jours"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PASTEL["violet_deep"]},
                    "steps": [
                        {"range": [0, 33], "color": PASTEL["green"]},
                        {"range": [33, 66], "color": PASTEL["orange"]},
                        {"range": [66, 100], "color": PASTEL["pink"]},
                    ],
                },
            ))
            _style_fig(fig, height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.metric("Statut réel observé (jeu synthétique)", "Défaut à 90 jours" if label_reel else "Pas de défaut")
            st.caption(
                "`defaut_90j` est l'étiquette du jeu de données synthétique (ce client a réellement "
                "fait défaut ou non dans la simulation) — à comparer, à titre indicatif, à la "
                "probabilité prédite par M3 ci-contre."
            )
    else:
        st.markdown("**Informations déclaratives** (les variables comportementales et le texte des "
                     "transactions, indisponibles pour un client hypothétique, sont fixées aux "
                     "moyennes réelles du portefeuille synthétique — éditables ci-dessous).")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Âge", min_value=18, max_value=35, value=26)
        sexe = c2.selectbox("Sexe", sorted(clients.sexe.unique()))
        zone = c3.selectbox("Zone", sorted(clients.zone.unique()))
        c4, c5, c6 = st.columns(3)
        secteur = c4.selectbox("Secteur", sorted(clients.secteur.unique()))
        region = c5.selectbox("Région", sorted(clients.region.unique()))
        anciennete = c6.number_input("Ancienneté bancaire (mois)", min_value=1, max_value=95, value=48)
        revenu = st.slider("Revenu déclaré (FCFA/mois)", min_value=30000, max_value=458000, value=147000, step=1000)

        moyennes_comp = df[COMPORTEMENTAL_NUM].mean()
        part_cols = [c for c in df.columns if c.startswith("PART_")]
        moyennes_parts = df[part_cols].mean()

        ligne_manuelle = pd.DataFrame([{
            "age": age, "sexe": sexe, "zone": zone, "secteur": secteur, "region": region,
            "revenu_declare": float(revenu), "anciennete_mois": anciennete,
            **moyennes_comp.to_dict(), **moyennes_parts.to_dict(), "texte": "",
        }])
        colonnes_attendues = [c for c in df.columns if c not in colonnes_gt + [TARGET, "client_id"]]
        X_manuel = ligne_manuelle[colonnes_attendues]
        proba = float(pipeline_m3.predict_proba(X_manuel)[:, 1][0])

        st.subheader("Score statistique (modèle M3)")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100, number={"suffix": " %"},
            title={"text": "Probabilité de défaut à 90 jours (estimation, variables comportementales = moyennes du portefeuille)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": PASTEL["violet_deep"]}},
        ))
        _style_fig(fig, height=280, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Score de l'agent LLM local (S1/S2/S3)")
    st.warning(
        "⚠️ Scoring LLM non disponible en direct dans cette démo — Ollama n'est pas démarré dans cet "
        "environnement. Ci-dessous, un exemple réel de dossier tel que transmis à l'agent LLM et les "
        "résultats déjà calculés du pilote (60 dossiers), extraits de `m5_m6_llm.ipynb`. Voir ce "
        "notebook pour rejouer le scoring sur un dossier précis."
    )
    with st.expander("Exemple réel de dossier transmis au LLM (premier dossier du pilote, m5_m6_llm.ipynb)"):
        st.code(EXEMPLE_DOSSIER_LLM, language=None)
    c1, c2, c3 = st.columns(3)
    for c, (nom, valeur) in zip((c1, c2, c3), ICC_LLM.items()):
        c.metric(f"ICC pilote — {nom}", f"{valeur:.3f}")
    st.caption(
        "Ces valeurs d'ICC proviennent du pilote de 20 dossiers rejoués (test-retest), pas de ce "
        "client précis — elles indiquent la fiabilité mesurée de l'agent LLM sur l'échantillon pilote."
    )


def page_equite_fiabilite():
    st.title("⚖️ Équité & fiabilité")

    st.subheader("Fiabilité test-retest de l'agent LLM (ICC)")
    st.caption(
        "ICC(C,1) — Koo & Mae (2016) — calculé sur 20 dossiers rejoués (`m5_m6_llm.ipynb`, Étape 4). "
        "Seuil de fiabilité usuel : 0,75 (ligne pointillée)."
    )
    icc_df = pd.DataFrame({"score": list(ICC_LLM.keys()), "icc": list(ICC_LLM.values())})
    fig = px.bar(icc_df, x="score", y="icc", range_y=[0, 1], text="icc",
                 color="icc", color_continuous_scale=[PASTEL["pink"], PASTEL["orange"], PASTEL["green"]],
                 range_color=[0, 1])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.add_hline(y=SEUIL_FIABILITE_ICC, line_dash="dash", line_color=PASTEL["muted"],
                  annotation_text="seuil de fiabilité 0,75", annotation_position="top left")
    _style_fig(fig, showlegend=False, coloraxis_showscale=False, yaxis_title="ICC(C,1)", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.error(
        "Aucun des trois scores n'atteint le seuil de 0,75 sur ce pilote réduit (60 dossiers, dont 20 "
        "rejoués). Ces valeurs proviennent de scores produits par l'ancien pipeline (température 0,7 "
        "dès le premier appel) ; la température de production par défaut est désormais 0,0 "
        "(déterministe), mais le test-retest n'a pas encore été rejoué avec ce nouveau défaut."
    )

    st.divider()
    st.subheader("Audit d'équité — secteur formel / informel (modèle M3, `equite.ipynb`)")
    st.caption(
        "Règle des 4/5 : un ratio d'approbation entre groupes < 0,80 signale un *disparate impact*. "
        "Correction A = seuils d'approbation différenciés par groupe (Hardt, Price & Srebro, 2016, "
        "version simplifiée) ; correction B = repondération des données (Kamiran & Calders, 2012)."
    )
    st.dataframe(
        EQUITE_SECTEUR.style.format({
            "auc_test": "{:.3f}", "ratio_4_5": "{:.3f}",
            "refus_vrais_bons_formel": "{:.1%}", "refus_vrais_bons_informel": "{:.1%}",
        }),
        use_container_width=True, hide_index=True,
    )
    fig2 = px.bar(EQUITE_SECTEUR, x="variante", y="ratio_4_5", text="ratio_4_5",
                  title="Ratio d'approbation formel/informel (règle des 4/5)",
                  color_discrete_sequence=[PASTEL["violet"]])
    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig2.add_hline(y=SEUIL_REGLE_4_5, line_dash="dash", line_color=PASTEL["muted"],
                   annotation_text="seuil des 4/5 (0,80)", annotation_position="top left")
    _style_fig(fig2, yaxis_title="ratio (min/max)", xaxis_title=None)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "La repondération (B) fait passer le ratio sous le seuil des 4/5 au coût d'une perte d'AUC "
        "(0,719 → 0,699) ; les seuils par groupe (A) corrigent le ratio sans perte d'AUC mais "
        "déplacent le seuil de décision par groupe, une approche différente sur le plan réglementaire."
    )

    st.subheader("Audit d'équité — sexe (modèle M3, `equite.ipynb`)")
    st.dataframe(EQUITE_SEXE.style.format({"ratio_4_5": "{:.2f}"}), use_container_width=True, hide_index=True)
    st.caption("Aucun signal de disparate impact par sexe au seuil des 4/5, avant ou après correction.")

    st.divider()
    st.subheader("Biais sectoriel documenté du score LLM S3 (non neutralisé)")
    st.warning(
        f"Le secteur déclaré reste visible du LLM dans le prompt utilisé pour produire S3 (risque "
        f"sectoriel). Sur l'échantillon pilote (audit du modèle M6) : ratio d'approbation par secteur "
        f"= **{BIAIS_S3_LLM['ratio_4_5_secteur']:.2f}** (disparate impact) ; refus des « vrais bons » : "
        f"**{BIAIS_S3_LLM['refus_vrais_bons_formel']:.1%}** (secteur formel) contre "
        f"**{BIAIS_S3_LLM['refus_vrais_bons_informel']:.1%}** (secteur informel). "
        "Recommandation du projet : sortir S3 de l'agent LLM et le brancher sur le modèle statistique "
        "M3 déjà corrigé pour l'équité (voir `llm_utils.py` et README.md)."
    )


def page_performance_modeles():
    st.title("📈 Performance des modèles (M0 → M6)")

    st.subheader("Échelle réelle du portefeuille (train 1 600 / test 400 clients)")
    st.caption("Résultats réels affichés par `baseline.ipynb`, `m2_m3_texte.ipynb` et `m4_embeddings.ipynb`.")
    _cmap_pastel_violet = LinearSegmentedColormap.from_list(
        "pastel_violet", [PASTEL["violet_soft"], PASTEL["violet_deep"]]
    )
    st.dataframe(
        PERF_PLEINE_ECHELLE.style.format({"auc": "{:.3f}", "gini": "{:.3f}", "ks": "{:.3f}"})
        .background_gradient(subset=["auc"], cmap=_cmap_pastel_violet),
        use_container_width=True, hide_index=True,
    )
    fig = px.bar(PERF_PLEINE_ECHELLE, x="modele", y="auc", text="auc", title="AUC par palier (pleine échelle)",
                 color_discrete_sequence=[PASTEL["blue"]])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    _style_fig(fig, yaxis_range=[0.5, 0.8], xaxis_title=None, yaxis_title="AUC (test)")
    fig.add_hline(y=0.5, line_dash="dot", line_color=PASTEL["muted"], annotation_text="hasard (AUC 0,5)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "M2 (catégorisation par lexique des libellés) apporte le principal gain statistiquement "
        "significatif (DeLong M1→M2 : p = 2,97e-03). Les gains M2→M3 (TF-IDF texte libre) et M1→M4 "
        "(embeddings CamemBERT) ne sont pas significatifs à cette échelle."
    )

    st.divider()
    st.subheader("Pilote de l'agent LLM (60 dossiers, 30 train / 30 test)")
    st.caption(
        "Apport des scores LLM (S1/S2/S3) en variables explicatives supplémentaires — README.md / "
        "`m5_m6_llm.ipynb`, cellule 21. Échantillon très réduit : à lire comme une démonstration du "
        "pipeline, pas une validation statistique (voir page Équité & fiabilité)."
    )
    st.dataframe(
        PERF_PILOTE_LLM.style.format({"auc": "{:.3f}", "gini": "{:.3f}", "ks": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )
    fig3 = px.bar(PERF_PILOTE_LLM, x="modele", y="auc", text="auc", title="AUC par palier (pilote LLM, 60 dossiers)",
                  color_discrete_sequence=[PASTEL["violet_deep"]])
    fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    _style_fig(fig3, xaxis_title=None, yaxis_title="AUC (test pilote)")
    fig3.add_hline(y=0.5, line_dash="dot", line_color=PASTEL["muted"], annotation_text="hasard (AUC 0,5)")
    st.plotly_chart(fig3, use_container_width=True)
    st.info(
        "L'apport des scores LLM (M5 vs M3, M6 vs M4) n'est pas statistiquement significatif au test "
        "de DeLong sur cet échantillon réduit (p = 1,00 et p = 0,28 respectivement) — voir README.md, "
        "section Résultats clés, pour le détail des recommandations avant tout usage en décision réelle."
    )


# ============================================================================
# Navigation
# ============================================================================

PAGES = {
    "Vue d'ensemble": page_vue_ensemble,
    "Scorer un client": page_scorer_client,
    "Équité & fiabilité": page_equite_fiabilite,
    "Performance des modèles": page_performance_modeles,
}

jl_logo_header()
st.sidebar.markdown("---")
st.sidebar.title("Afri — Navigation")
choix = st.sidebar.radio("Page", list(PAGES.keys()))
st.sidebar.divider()
st.sidebar.caption(
    "Projet de scoring crédit comportemental M0-M6, auteur K. Jessy (stage BEAC / protocole "
    "TOFE-Congo, ISSEA). Ce dashboard affiche uniquement des données et résultats déjà produits "
    "par le projet — voir README.md pour le détail méthodologique complet."
)

PAGES[choix]()
