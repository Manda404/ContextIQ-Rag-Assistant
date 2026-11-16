# ContextIQ-Rag-Assistant

Un assistant documentaire en Python pour **retrouver des informations, produire des réponses sourcées et évaluer leur fiabilité**. Le projet compare trois méthodes de recherche sur une même base de connaissances : **TF-IDF, BM25 et embeddings multilingues**.

Les notebooks présentent les expériences et leurs résultats. La logique métier est regroupée dans des classes Python réutilisables, avec validation des données et des configurations par **Pydantic**.

Pour comprendre la démarche, les choix techniques et les résultats métier, consultez le [rapport de présentation du projet](README-Report-RAG.md).

## Fonctionnement

```text
Documents JSON → Validation → Index de recherche
                                      ↓
Question → Recherche → Décision d’abstention → Contrôles de sécurité
                                                      ↓
                                           Extrait cité ou refus
```

Le système restitue actuellement le meilleur document retenu, accompagné de son identifiant. Il prépare aussi un prompt pour une future génération par LLM, mais **aucun LLM n’est appelé**.

L’évaluation couvre :

- **Recherche** : Precision@K, Recall@K, Hit Rate@K et MRR@K.
- **Abstention** : faux refus et acceptations de questions hors périmètre.
- **Réponses** : pertinence de la source, fidélité de l’extrait, validité des citations et refus corrects.
- **Sécurité** : détection heuristique d’injections de prompt dans les questions et les documents.

## Installation

Prérequis : **Python 3.15.2** et **Poetry 2.1.3**. Depuis la racine du projet :

```sh
poetry env use python3.15
poetry install
make ci
```

Dans votre éditeur, ouvrez les notebooks avec l’environnement Python créé par Poetry. L’utilisation de TF-IDF, de BM25 et des réponses extractives ne nécessite aucune clé d’API.

### Données

Placez les deux fichiers UTF-8 suivants dans `data/raw/`, ou fournissez votre propre dossier au chargeur. Les exemples ci-dessous illustrent le format attendu et ne reproduisent pas les données du projet.

**`knowledge_base.json`** — liste de documents aux identifiants uniques :

```json
[
  {"id": "DOC-1", "text": "Le support est ouvert du lundi au vendredi."}
]
```

**`eval_set.json`** — questions et identifiants des documents attendus :

```json
[
  {"q": "Quand le support est-il ouvert ?", "gold_docs": ["DOC-1"]},
  {"q": "Quel temps fera-t-il demain ?", "gold_docs": []}
]
```

Une liste `gold_docs` vide désigne une question hors périmètre. Les références sont vérifiées au chargement. Le dossier `data/` est exclu de Git.

### Modèle d’embeddings

Le modèle utilisé est `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Son nom et sa révision sont définis dans [configs/retrieval.yaml](configs/retrieval.yaml).

Par défaut, `local_files_only: true` impose l’utilisation de poids déjà présents dans le cache local. Pour une première utilisation sans cache, passez temporairement cette valeur à `false`, puis initialisez `EmbeddingRetriever` dans le notebook 02. Cette étape télécharge les poids et nécessite une connexion Internet. Remettez ensuite la valeur à `true` pour les exécutions hors ligne.

Les poids sont conservés dans le cache utilisateur Hugging Face, ou dans le dossier indiqué par `cache_dir`. Les vecteurs des documents sont calculés à l’initialisation du retriever et conservés en mémoire.

## Plateforme interactive

Pour utiliser le projet sans écrire de code :

```sh
make app
```

Ouvrez **http://127.0.0.1:8501**. L’interface reprend la palette verte et les thèmes clair/sombre du dashboard BRVM, avec cinq espaces :

- **Simulation** : poser une question, choisir la méthode, K et le seuil ; lire l’extrait cité ou le refus et comprendre les étapes du traitement.
- **Base documentaire** : consulter et filtrer les sources, puis examiner leur couverture dans le jeu d’évaluation.
- **Évaluations** : comparer Precision@K, Recall@K, Hit Rate@K et MRR@K par méthode dans un tableau et un graphique, puis examiner séparément les réponses et les refus. Les résultats sont enregistrés dans MLflow.
- **Suivi des conversations** : relire les questions et réponses persistées, inspecter les sources et signaler les échanges hors sujet, incorrects ou incomplets.
- **Comprendre le projet** : découvrir le pipeline, le prompt, les limites et les rapports sans quitter la plateforme.

La présentation ressemble à un échange, mais chaque question est traitée indépendamment : **aucun LLM ni mémoire conversationnelle n’est utilisé**. Les échanges sont journalisés localement pour analyse ; les textes ne sont pas enregistrés dans MLflow. Les modèles sont chargés à la demande. Les changements de configuration ou de données recréent les ressources et effacent l’historique visible de simulation, en conservant le journal persistant.

L’interface écoute uniquement sur la machine locale. TF-IDF et BM25 permettent de commencer sans télécharger de modèle. Arrêtez le serveur avec Ctrl+C.

## Utilisation rapide

Depuis la racine du projet ou le dossier `notebooks/`, cet exemple utilise BM25 sans télécharger de modèle :

```python
from contextiq.generation.extractive_answer import ExtractiveAnswerGenerator
from contextiq.ingestion.loader import DataLoader
from contextiq.retrieval.bm25 import BM25Retriever
from contextiq.safety.abstention import AbstentionPolicy
from contextiq.settings import Settings

settings = Settings.load()
dataset = DataLoader().load()

retriever = BM25Retriever(
    dataset.documents,
    **settings.retrieval.bm25.model_dump(),
)
policy = AbstentionPolicy(
    threshold=settings.retrieval.abstention_thresholds["BM25"],
)

question = "Quand le support est-il ouvert ?"  # À adapter à votre base.
results = retriever.retrieve(question, top_k=settings.retrieval.top_k)
decision = policy.decide(results)
answer = ExtractiveAnswerGenerator().generate(question, decision)

print(answer.text)
```

Pour utiliser des données situées ailleurs :

```python
dataset = DataLoader(data_dir="chemin/vers/mes-donnees").load()
```

Les chemins du projet sont détectés automatiquement depuis sa racine ou ses notebooks. Aucun chemin propre à une machine n’est nécessaire. La commande `contextiq` fournit actuellement uniquement l’aide et la version ; les expériences s’exécutent via les notebooks ou les classes Python.

## Parcours des notebooks

À consulter dans cet ordre ; chaque notebook initialise les objets dont il a besoin.

| Notebook | Objectif |
| --- | --- |
| [01 — Comprendre les données](notebooks/01_data_understanding.ipynb) | Examiner les documents, les questions et la couverture des annotations. |
| [02 — Expérimenter la recherche](notebooks/02_retrieval_experiments.ipynb) | Comprendre le prétraitement et comparer les résultats des trois méthodes. |
| [03 — Évaluer la recherche](notebooks/03_retrieval_evaluation.ipynb) | Comparer les métriques pour plusieurs valeurs de K et inspecter les erreurs. |
| [04 — Analyser l’abstention](notebooks/04_abstention_analysis.ipynb) | Explorer les seuils, les faux refus et les acceptations hors périmètre. |
| [05 — Prompt et sécurité](notebooks/05_prompt_and_safety.ipynb) | Construire un contexte sourcé et examiner les contrôles contre les injections. |
| [06 — Produire des réponses](notebooks/06_extractive_answers.ipynb) | Restituer un extrait cité ou refuser de répondre. |
| [07 — Évaluer les réponses](notebooks/07_answer_evaluation.ipynb) | Analyser des réponses métier et mesurer la qualité du pipeline extractif. |
| [08 — Suivre avec MLflow](notebooks/08_mlflow_monitoring.ipynb) | Enregistrer et comparer les paramètres et métriques des évaluations locales. |

## Configuration et logs

Les paramètres sont chargés par `Settings.load()` et validés avec Pydantic.

| Fichier | Paramètres |
| --- | --- |
| [retrieval.yaml](configs/retrieval.yaml) | Top K, BM25, modèle d’embeddings et seuils d’abstention par méthode. |
| [evaluation.yaml](configs/evaluation.yaml) | Valeurs de K et grilles de seuils à comparer. |
| [logging.yaml](configs/logging.yaml) | Affichage des logs, dossier, niveau, rotation et rétention. |
| [monitoring.yaml](configs/monitoring.yaml) | Expérience MLflow, dossier local de stockage et port de l’interface. |

La journalisation **Loguru** écrit dans `logs/contextiq-rag-assistant.log`. L’affichage dans les notebooks et le terminal est **désactivé par défaut** :

```yaml
console: false
log_dir: logs
level: INFO
rotation: "10 MB"
retention: "7 days"
```

Passez `console` à `true` pour afficher aussi les logs, puis rechargez `Settings.load()`. Les traces automatiques enregistrent les opérations, durées et métadonnées techniques sans inclure le texte des questions, documents ou réponses. Les fichiers de logs sont exclus de Git.

Pour déplacer les données ou les configurations, utilisez `CONTEXTIQ_DATA_DIR` ou `CONTEXTIQ_CONFIG_DIR`, ou fournissez `data_dir` et `config_dir` aux classes concernées. Les variables doivent être exportées ou définies dans l’environnement Python : **le fichier `.env` n’est pas chargé automatiquement**. Les options disponibles figurent dans [.env.example](.env.example).

## Suivi avec MLflow

Exécutez le [notebook 08](notebooks/08_mlflow_monitoring.ipynb) pour enregistrer une évaluation par méthode. Pour consulter et comparer les exécutions :

```sh
make mlflow-ui
```

Ouvrez **http://127.0.0.1:5000**, puis sélectionnez l’expérience `contextiq-rag-assistant`. Le port est configurable dans `configs/monitoring.yaml` ; Ctrl+C arrête le serveur.

MLflow conserve les paramètres, les métriques agrégées, les versions et les empreintes des données dans `mlflow/`, exclu de Git. Les textes métier ne sont pas enregistrés et la télémétrie est désactivée. Le serveur est local et n’est pas nécessaire pour enregistrer les évaluations.

Le [plan de suivi en production](reports/production_monitoring.md) décrit les indicateurs, les alertes à prévoir, les retours conseillers et les vérifications après modification documentaire. La collecte en continu et les notifications automatiques ne sont pas encore implémentées.

### Journal des questions et réponses

Chaque question envoyée dans la simulation est enregistrée automatiquement dans **`logs/conversations.db`**, une base SQLite locale. Le journal conserve la question, la réponse ou le refus, les sources retrouvées et citées, les scores, les paramètres, la durée, les identifiants d’échange et de session, ainsi que les empreintes du corpus, de la configuration et du prompt. Les erreurs techniques sont enregistrées avec leur type.

La page **Suivi des conversations** permet de consulter les derniers échanges, de les filtrer par méthode et d’enregistrer une revue humaine : utile, hors sujet, incorrecte ou incomplète. Les compteurs couvrent tout le journal ; les taux de signalement portent uniquement sur les échanges revus. Ils ne constituent pas une mesure automatique de pertinence sur tout le trafic.

Le bouton d’enregistrement du bilan crée un run dans l’expérience MLflow locale `contextiq-rag-assistant-conversations`, avec **les agrégats uniquement**. Les textes et commentaires restent dans le journal SQLite. Les rafraîchissements et la consultation des échanges ne créent pas de nouveaux enregistrements.

Cette journalisation est activée par `log_conversations: true` dans `configs/monitoring.yaml`. La passer à `false` arrête les nouveaux enregistrements et masque la page de consultation ; les anciennes données ne sont pas supprimées. Le dossier suit `log_dir` de `configs/logging.yaml`, ou `CONTEXTIQ_LOG_DIR`. `console: false` continue à masquer les traces dans les notebooks et le terminal.

Le fichier Loguru `logs/contextiq-rag-assistant.log` conserve les traces techniques et les identifiants de corrélation, sans recopier les textes des échanges. Les deux fichiers sont exclus de Git. Le journal SQLite contient du texte métier et n’a pas de purge automatique ; sa conservation est distincte de la rotation des logs techniques.

## Organisation du projet

```text
ContextIQ-Rag-Assistant/
├── configs/           # Paramètres YAML
├── data/raw/          # Données locales, exclues de Git
├── src/contextiq/
│   ├── ingestion/     # Chargement et prétraitement
│   ├── retrieval/     # TF-IDF, BM25 et embeddings
│   ├── generation/    # Prompt et réponses extractives
│   ├── evaluation/    # Métriques et évaluateurs
│   ├── safety/        # Abstention et détection d’injections
│   ├── monitoring/    # Évaluations et suivi local MLflow
│   ├── ui/            # Interface Streamlit et service de simulation
│   ├── schemas/       # Modèles Pydantic
│   ├── settings.py    # Chargement des configurations
│   └── logging.py     # Journalisation Loguru
├── notebooks/         # Parcours d’expérimentation et d’analyse
├── tests/             # Tests unitaires
├── reports/           # Protocole, résultats et analyse des erreurs
├── logs/              # Journaux locaux, exclus de Git
└── .github/workflows/ # Pipeline CI
```

## Qualité du code

```sh
make app           # Plateforme locale sur le port 8501
make ci            # Validation Poetry, lint, formatage et tests unitaires
make test          # Tests seuls, sans téléchargement de modèle
make lint          # Analyse statique avec Ruff
make format-check  # Vérification du formatage
make format        # Application du formatage
```

Le workflow GitHub Actions s’exécute à chaque push, pull request ou lancement manuel. Il sépare deux jobs : qualité/formatage et construction du package sur Ubuntu ; tests et vérification de la commande installée sur macOS avec Python 3.15.2. Poetry est isolé avec pipx et les environnements sont mis en cache à partir du lockfile. Les tests ne nécessitent ni données métier ni téléchargement de modèle. La télémétrie est désactivée et les logs de CI restent dans le dossier temporaire du runner.

Ces tests vérifient le comportement du code ; les notebooks évaluent la qualité des résultats sur les données métier. Aucun déploiement automatique n’est configuré.

## Résultats et limites

Le [rapport d’évaluation](reports/evaluation_report.md) présente les mesures obtenues et les erreurs observées. Le [protocole d’évaluation](reports/evaluation_protocol.md) décrit les critères, les jeux à constituer et les validations à réaliser. Les résultats détaillés sont disponibles dans le [snapshot des métriques](reports/metrics/evaluation_snapshot.json).

Les résultats actuels sont **exploratoires**, sur six documents et six questions. Les seuils d’abstention sont illustratifs, sans calibration sur un jeu indépendant. Une source citée peut rester non pertinente : la fidélité d’un extrait ne suffit pas à garantir une réponse correcte. La détection d’injections est heuristique et ne constitue pas une protection complète.
