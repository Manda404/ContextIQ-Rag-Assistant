"""Plateforme Streamlit locale : simulation, exploration et évaluations du RAG."""

from pathlib import Path

import streamlit as st

from contextiq.generation.prompt_builder import PromptBuilder
from contextiq.ingestion.loader import DataLoader
from contextiq.monitoring.conversations import ConversationFeedback
from contextiq.monitoring.mlflow_tracking import MLflowTracker, fingerprint
from contextiq.settings import Settings
from contextiq.ui.service import AssistantSession, SimulationTurn
from contextiq.ui.theme import apply_theme, header

STARTER_QUESTIONS = (
    "Combien de temps de préavis pour un contrat mensuel ?",
    "Quels sont les frais de résiliation d’un contrat annuel avant son terme ?",
    "Quels frais faut-il régler pour la réactivation d’une ligne suspendue pour retard de paiement ?",
    "Quelles sont les conditions d’ancienneté et de paiement pour proposer une offre de rétention ?",
)

REASONS = {
    "no_results": "Aucun document retrouvé",
    "no_positive_score": "Aucun score positif",
    "below_threshold": "Meilleur score inférieur au seuil",
    "threshold_met": "Seuil de recherche atteint",
    "abstention": "Contexte jugé insuffisant par la règle d’abstention",
    "suspicious_question": "Instruction suspecte dans la question",
    "suspicious_context": "Instruction suspecte dans une source",
    "empty_context": "Aucun contexte exploitable",
    "extractive": "Extrait original accompagné de sa source",
}


def get_session() -> AssistantSession:
    """Actualiser les ressources si les données ou les configurations ont changé."""
    settings = Settings.load()
    dataset = DataLoader().load()
    signature = fingerprint({"data": dataset.model_dump(), "settings": settings.model_dump()})
    if st.session_state.get("signature") != signature:
        st.session_state.engine = AssistantSession(dataset, settings)
        st.session_state.signature = signature
        st.session_state.turns = []
        st.session_state.evaluations = []
    return st.session_state.engine


def render_turn(turn: SimulationTurn) -> None:
    """Afficher le résultat puis son raisonnement technique, sans inventer d'explication métier."""
    with st.chat_message("user"):
        st.text(turn.answer.question)
    with st.chat_message("assistant"):
        if turn.answer.answered:
            st.caption("EXTRAIT DOCUMENTAIRE · SOURCE CITÉE")
        else:
            st.warning("Réponse refusée · " + REASONS.get(turn.answer.reason, turn.answer.reason))
        st.text(turn.answer.text)
        if turn.interaction_id:
            st.caption(f"Échange enregistré · {turn.interaction_id[:12]}")
        st.caption(
            f"{turn.method} · K={turn.top_k} · seuil {turn.decision.threshold:.3f} · "
            f"{turn.elapsed_ms:.0f} ms (chargement éventuel inclus)"
        )
        with st.expander("Comprendre ce résultat"):
            score = turn.decision.best_score
            st.write("**1. Recherche** —", f"{len(turn.decision.results)} document(s) retrouvé(s).")
            st.write("**2. Abstention** —", REASONS[turn.decision.reason])
            st.write("Meilleur score :", f"{score:.3f}" if score is not None else "Aucun")
            st.write("**3. Sécurité et réponse** —", REASONS[turn.answer.reason])
            if turn.answer.blocked_source_ids:
                st.caption("Le contenu suspect est masqué. Aucune source bloquée n’est restituée.")
            else:
                for rank, result in enumerate(turn.decision.results, 1):
                    st.text(f"{rank}. Source {result.document.id} · score {result.score:.3f}")
                    st.text(result.document.text)
            if turn.prompt.ready:
                st.write("**4. Prompt préparé** — consultable, sans envoi à un LLM.")
                for message in turn.prompt.messages:
                    st.caption(message.role.upper())
                    st.code(message.content, language="json" if message.role == "user" else None)


def simulation(session: AssistantSession) -> None:
    """Proposer un échange visuel dont chaque question est traitée indépendamment."""
    st.subheader("Simuler une question conseiller")
    st.caption(
        "Une présentation conversationnelle, des réponses extractives : aucun LLM n’est appelé. "
        "Chaque question est indépendante ; les échanges précédents ne sont pas utilisés."
    )
    left, middle, right = st.columns([2, 1, 1])
    method = left.selectbox("Méthode de recherche", ["TF-IDF", "BM25", "Embeddings"], key="method")
    k = middle.number_input(
        "Documents recherchés (K)",
        min_value=1,
        max_value=max(3, len(session.dataset.documents), session.settings.retrieval.top_k),
        value=session.settings.retrieval.top_k,
        step=1,
    )
    threshold = right.number_input(
        "Seuil d’abstention",
        min_value=0.0,
        value=session.settings.retrieval.abstention_thresholds[method],
        step=0.05,
        format="%.3f",
        key=f"threshold_{method}",
    )
    st.caption(
        "Les scores ne sont pas des probabilités. Les seuils sont exploratoires et propres à chaque méthode."
    )
    with st.expander("Des questions pour commencer"):
        for number, suggestion in enumerate(STARTER_QUESTIONS, start=1):
            st.text(f"{number}. {suggestion}")
    if st.button("Effacer les échanges de cette session", key="clear_history"):
        st.session_state.turns = []
    st.caption(
        "Questions et réponses enregistrées localement pour la revue de qualité. "
        "Effacer la session ne supprime pas le journal."
        if session.conversations is not None
        else "Journal des conversations désactivé dans la configuration."
    )
    for turn in st.session_state.turns:
        render_turn(turn)
    question = st.chat_input("Posez une question sur la base documentaire…", max_chars=2000)
    if question:
        try:
            with st.spinner("Recherche des sources et vérification du contexte…"):
                turn = session.ask(question, method, int(k), threshold)
            # Historique limité à la session, sans écriture des textes dans MLflow.
            st.session_state.turns = [*st.session_state.turns[-19:], turn]
            render_turn(turn)
        except Exception:
            st.error(
                "La recherche n’a pas abouti. Vérifiez la longueur du texte et les fichiers du modèle. "
                "Pour les embeddings, les poids doivent être disponibles dans le cache local. "
                "TF-IDF et BM25 fonctionnent sans ces poids. Vérifiez aussi l’accès au journal local."
            )


def documents(session: AssistantSession) -> None:
    """Explorer les sources et leur couverture, séparément des réponses de la simulation."""
    st.subheader("Comprendre la base documentaire")
    st.write(
        "Les documents sont conservés dans leur forme originale et cités par leur identifiant."
    )
    query = st.text_input("Filtrer les documents", placeholder="Un identifiant ou un mot du texte")
    for document in session.dataset.documents:
        if query.casefold() not in (document.id + " " + document.text).casefold():
            continue
        with st.container(border=True):
            st.text(document.id)
            st.text(document.text)
            count = sum(document.id in q.gold_docs for q in session.dataset.questions)
            st.caption(f"Questions d’évaluation associées : {count}")
    with st.expander("Questions annotées : comprendre le jeu d’évaluation"):
        st.caption("Ces références servent à mesurer la qualité, pas à choisir les réponses.")
        for number, question in enumerate(session.dataset.questions, 1):
            st.text(f"{number}. {question.q}")
            st.text("Sources attendues : " + (", ".join(question.gold_docs) or "Hors périmètre"))


def evaluations(session: AssistantSession) -> None:
    """Évaluer uniquement sur action explicite ; un changement de thème ne crée aucun run."""
    st.subheader("Comparer et suivre les évaluations")
    st.info(
        "Mesures exploratoires sur le jeu fourni. Une copie fidèle peut être hors sujet. "
        "Les seuils utilisés ici viennent des configurations, indépendamment des essais de simulation."
    )
    methods = st.multiselect(
        "Méthodes à évaluer",
        ["TF-IDF", "BM25", "Embeddings"],
        default=["TF-IDF", "BM25", "Embeddings"],
    )
    k = st.selectbox("K pour l’évaluation", session.settings.evaluation.k_values)
    if st.button("Évaluer et enregistrer dans MLflow", type="primary", disabled=not methods):
        for method in methods:
            try:
                with st.spinner(f"Évaluation de {method}…"):
                    run = session.evaluate(method, k)
                st.session_state.evaluations.append({"method": method, "k": k, "run": run})
            except Exception:
                st.error(
                    f"Évaluation de {method} impossible. Vérifiez le modèle et le stockage local."
                )
    with st.expander("Comprendre les métriques de recherche", expanded=True):
        st.markdown(
            "- **Precision@K** : nombre de sources pertinentes retrouvées, divisé par K.\n"
            "- **Recall@K** : proportion des sources attendues présentes dans les K premiers résultats.\n"
            "- **Hit Rate@K** : proportion des questions ayant au moins une bonne source dans le top K.\n"
            "- **MRR@K — Mean Reciprocal Rank** : moyenne de l’inverse du rang de la première "
            "bonne source ; 1 au premier rang, 0,5 au deuxième, 0 si absente du top K."
        )
        st.caption(
            "Scores entre 0 et 1 : plus ils sont élevés, mieux c’est. Calcul avant abstention, "
            "uniquement sur les questions avec une source attendue. Precision@K conserve K "
            "au dénominateur même si moins de K documents sont retournés."
        )
    if st.session_state.evaluations:
        # Garder le dernier résultat par méthode et K, sans supprimer l'historique MLflow.
        latest = {(item["method"], item["k"]): item for item in st.session_state.evaluations}
        metric_names = {
            "Precision@K": "retrieval_precision",
            "Recall@K": "retrieval_recall",
            "Hit Rate@K": "retrieval_hit_rate",
            "MRR@K": "retrieval_mrr",
        }
        retrieval_rows, answer_rows = [], []
        for item in latest.values():
            metrics = item["run"].metrics
            retrieval_rows.append(
                {
                    "Méthode": item["method"],
                    "K": item["k"],
                    "Questions avec source": int(
                        metrics.get(
                            "answer_in_scope_count",
                            sum(bool(q.gold_docs) for q in session.dataset.questions),
                        )
                    ),
                    **{label: metrics.get(key) for label, key in metric_names.items()},
                }
            )
            answer_rows.append(
                {
                    "Méthode": item["method"],
                    "K": item["k"],
                    "Questions": int(metrics["answer_question_count"]),
                    "Succès conjoint": metrics["answer_success_rate"],
                    "Faux refus": metrics.get("answer_false_refusal_rate"),
                    "Fausses acceptations": metrics.get("abstention_false_acceptance_rate"),
                    "Run MLflow": item["run"].run_id,
                }
            )
        st.subheader("1. Qualité du retrieval — avant abstention")
        st.dataframe(
            retrieval_rows,
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.3f") for name in metric_names
            },
        )
        st.caption(
            "Chaque ligne compare une méthode à une valeur de K. Les questions hors périmètre "
            "sont exclues de ces moyennes. Avec une seule source attendue par question, "
            "Recall@K et Hit Rate@K sont identiques."
        )
        current = {
            row["Méthode"]: row
            for row in retrieval_rows
            if row["K"] == k and row["Méthode"] in methods
        }
        # Le graphique compare seulement les méthodes évaluées au même K sélectionné.
        if current and any(
            row[name] is not None for row in current.values() for name in metric_names
        ):
            st.write(f"**Comparaison des méthodes à K={k}**")
            chart_rows = [
                {"Métrique": label, **{method: row[label] for method, row in current.items()}}
                for label in metric_names
            ]
            palette = {"TF-IDF": "#47751b", "BM25": "#56866c", "Embeddings": "#387a99"}
            st.bar_chart(
                chart_rows,
                x="Métrique",
                y=list(current),
                stack=False,
                sort=False,
                color=[palette[method] for method in current],
                y_label="Score (0 à 1)",
            )
        st.subheader("2. Qualité des réponses et décisions de refus")
        st.dataframe(
            answer_rows,
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.3f")
                for name in ("Succès conjoint", "Faux refus", "Fausses acceptations")
            },
        )
        st.caption(
            "Ces taux mesurent le pipeline après abstention et sécurité. Succès : toutes les questions ; "
            "faux refus : questions avec source ; fausses acceptations : hors périmètre. "
            "Une valeur absente indique un groupe non représenté, pas un score nul."
        )
    else:
        st.info(
            "Choisissez les méthodes et K, puis cliquez sur « Évaluer et enregistrer dans MLflow » pour afficher les scores calculés sur votre jeu de questions."
        )
    st.write("**Consulter l’historique complet**")
    st.code("make mlflow-ui", language="bash")
    st.link_button("Ouvrir MLflow local", f"http://127.0.0.1:{session.settings.monitoring.port}")
    st.caption(
        "L’interface MLflow doit être démarrée séparément. Aucun texte métier n’y est enregistré."
    )


def conversation_monitoring(session: AssistantSession) -> None:
    """Relire les échanges persistés et recueillir un verdict humain sur leur utilité."""
    st.subheader("Suivre les conversations réelles")
    store = session.conversations
    if store is None:
        st.info("Le journal est désactivé. Activez log_conversations dans configs/monitoring.yaml.")
        return
    if st.session_state.pop("review_saved", False):
        st.success("Revue enregistrée. Les indicateurs ont été actualisés.")
    summary = store.summarize()
    first, second, third, fourth = st.columns(4)
    first.metric("Requêtes enregistrées", summary.total)
    second.metric("Réponses refusées", summary.refused)
    third.metric("Erreurs techniques", summary.errors)
    fourth.metric("Échanges signalés / revus", f"{summary.flagged} / {summary.reviewed}")
    st.caption(
        "Effectifs sur tout le journal local, toutes sessions et versions confondues. "
        "Un refus n’est pas automatiquement une erreur. Le taux de signalement porte "
        "uniquement sur les échanges revus : il ne mesure pas la précision globale du chatbot."
    )
    records = store.recent(limit=200)
    if not records:
        st.info("Posez une question dans Simulation pour commencer à alimenter ce journal.")
        return
    method = st.selectbox("Filtrer par méthode", ["Toutes", "TF-IDF", "BM25", "Embeddings"])
    records = [record for record in records if method == "Toutes" or record.method == method]
    if not records:
        st.info("Aucun échange récent pour cette méthode.")
        return
    st.caption(
        "Les 200 derniers échanges au maximum sont chargés avant filtrage. Dates affichées en UTC."
    )
    labels = {"answered": "Réponse produite", "refused": "Refus", "error": "Erreur technique"}
    st.dataframe(
        [
            {
                "Date (UTC)": record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Méthode": record.method,
                "Question": record.question,
                "Statut": labels[record.status],
                "Durée (ms)": round(record.elapsed_ms),
                "Identifiant": record.id,
            }
            for record in records
        ],
        hide_index=True,
        width="stretch",
    )
    record_by_id = {record.id: record for record in records}
    selected = st.selectbox(
        "Inspecter un échange",
        list(record_by_id),
        format_func=lambda key: (
            f"{record_by_id[key].created_at:%Y-%m-%d %H:%M:%S} UTC · "
            f"{record_by_id[key].method} · {key[:12]}"
        ),
    )
    record = record_by_id[selected]
    with st.container(border=True):
        st.write("**Question**")
        st.text(record.question)
        st.write("**Réponse enregistrée**")
        st.text(record.answer or "Aucune réponse produite : erreur technique.")
        st.caption(f"{labels[record.status]} · K={record.top_k} · seuil {record.threshold:.3f}")
        if record.error_type:
            st.text("Type d’erreur : " + record.error_type)
        elif record.reason:
            st.text("Motif : " + REASONS.get(record.reason, record.reason))
        with st.expander("Sources et traçabilité de cet échange"):
            st.text("Session : " + record.session_id)
            st.text("Empreinte du corpus : " + record.corpus_sha256)
            st.text("Empreinte de configuration : " + record.configuration_sha256)
            st.text("Empreinte du prompt : " + record.prompt_sha256)
            st.text("Sources citées : " + (", ".join(record.cited_source_ids) or "Aucune"))
            for result in record.retrieved_sources:
                st.text(f"Source {result.document.id} · score {result.score:.3f}")
                st.text(result.document.text)
    verdicts = {
        "Utile et correcte": "helpful",
        "Hors sujet": "irrelevant",
        "Incorrecte": "incorrect",
        "Incomplète": "incomplete",
    }
    previous = store.get_feedback(selected)
    choices = list(verdicts)
    index = list(verdicts.values()).index(previous.verdict) if previous else 0
    with st.form(f"review_{selected}"):
        st.write("**Évaluer cet échange**")
        verdict = st.selectbox("Votre verdict", choices, index=index, key=f"verdict_{selected}")
        comment = st.text_area(
            "Commentaire de revue",
            value=previous.comment if previous else "",
            max_chars=2000,
            key=f"review_comment_{selected}",
        )
        if st.form_submit_button("Enregistrer la revue"):
            store.review(
                ConversationFeedback(
                    interaction_id=selected,
                    verdict=verdicts[verdict],
                    comment=comment,
                )
            )
            st.session_state.review_saved = True
            st.rerun()
    if st.button("Enregistrer le bilan des conversations dans MLflow"):
        try:
            run_id = MLflowTracker(session.settings).log_conversation_summary(store.summarize())
            st.success(f"Bilan enregistré · run {run_id}")
        except Exception:
            st.error("Enregistrement MLflow impossible. Vérifiez le stockage local.")
    st.caption(
        "Seuls les effectifs et taux agrégés sont envoyés à l’expérience MLflow locale "
        "« conversations ». Les questions, réponses et commentaires restent dans logs/conversations.db. "
        "Le suivi est continu pour les requêtes ; les verdicts de pertinence nécessitent une revue humaine."
    )


def project_guide() -> None:
    """Permettre de comprendre le projet et ses limites sans quitter la plateforme."""
    st.subheader("Le projet, étape par étape")
    st.write("**1. Charger** — les documents et annotations sont validés avec Pydantic.")
    st.write(
        "**2. Rechercher** — TF-IDF et BM25 comparent les mots ; les embeddings comparent des représentations sémantiques."
    )
    st.write(
        "**3. Décider** — un seuil propre à la méthode accepte ou refuse le contexte ; il ne garantit pas sa pertinence."
    )
    st.write(
        "**4. Protéger** — un filtre repère certaines instructions suspectes avant restitution."
    )
    st.write(
        "**5. Répondre** — le meilleur document autorisé est copié et cité, sans synthèse ni raisonnement client."
    )
    st.write(
        "**6. Évaluer** — sources attendues, fidélité, citations et refus sont mesurés ; MLflow conserve les agrégats."
    )
    st.warning(
        "Limites : jeu réduit, seuils non calibrés indépendamment, filtre de sécurité heuristique. Une source citée peut être incorrecte pour la question."
    )
    with st.expander("Lire le prompt système"):
        st.code(PromptBuilder.SYSTEM_PROMPT, language=None)
    root = Path(__file__).resolve().parents[3]
    choices = {
        "Rapport d’évaluation": "evaluation_report.md",
        "Protocole d’évaluation": "evaluation_protocol.md",
        "Suivi en production": "production_monitoring.md",
    }
    selection = st.selectbox("Lire un document du projet", list(choices))
    path = root / "reports" / choices[selection]
    if path.exists():
        # Ces rapports sont des fichiers du projet ; aucun HTML brut n'est autorisé.
        st.markdown(path.read_text(encoding="utf-8"), unsafe_allow_html=False)


def main() -> None:
    """Présenter une navigation locale et garder chaque simulation dans sa session."""
    st.set_page_config(page_title="ContextIQ · Document Intelligence", page_icon="◈", layout="wide")
    with st.sidebar:
        st.title("◈ ContextIQ")
        st.caption("DOCUMENT INTELLIGENCE")
        page = st.radio(
            "Explorer",
            [
                "Simulation",
                "Base documentaire",
                "Évaluations",
                "Suivi des conversations",
                "Comprendre le projet",
            ],
        )
        dark = st.toggle("Thème sombre", value=False)
        st.divider()
        st.caption("Réponses sourcées · Exécution locale · Sans appel LLM")
    apply_theme(dark)
    header()
    try:
        session = get_session()
    except Exception:
        st.error(
            "Impossible de charger le projet. Vérifiez les configurations YAML et les deux fichiers JSON de données."
        )
        st.stop()
    first, second, third = st.columns(3)
    first.metric("Documents", len(session.dataset.documents))
    second.metric("Méthodes de recherche", 3)
    third.metric("Questions d’évaluation", len(session.dataset.questions))
    if page == "Simulation":
        simulation(session)
    elif page == "Base documentaire":
        documents(session)
    elif page == "Évaluations":
        evaluations(session)
    elif page == "Suivi des conversations":
        conversation_monitoring(session)
    else:
        project_guide()
    st.html(
        '<div class="footer">ContextIQ · Exploration, sources et évaluation · Prototype expérimental</div>'
    )


if __name__ == "__main__":
    main()
