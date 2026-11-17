# ContextIQ-Rag-Assistant

ContextIQ est un assistant documentaire destiné aux conseillers clientèle. Il retrouve des informations dans une base de connaissances, restitue un extrait avec sa source et peut refuser de répondre lorsque le contexte est jugé insuffisant.

Le projet compare **TF-IDF, BM25 et les embeddings multilingues**. Il permet d’évaluer la recherche, les réponses et les décisions de refus. **Aucun LLM n’est appelé** : les réponses reprennent les documents originaux.

## Par où commencer ?

| Votre objectif | Où aller |
| --- | --- |
| Comprendre la démarche et les choix | [Rapport de présentation](README-Report-RAG.md) |
| Vous repérer dans le code et les fichiers | [Structure du projet](#structure-du-projet) |
| Essayer l’assistant sans écrire de code | [Application Streamlit](#utiliser-lapplication) |
| Suivre les expériences pas à pas | [Notebooks](#parcourir-les-notebooks) |
| Consulter les scores et les erreurs | [Résultats et rapports](#retrouver-les-résultats-et-les-rapports) |
| Modifier les paramètres | [Configuration](#adapter-la-configuration) |

## Structure du projet

```text
ContextIQ-Rag-Assistant/
├── README.md                         # Guide de navigation et d’utilisation
├── README-Report-RAG.md              # Démarche, choix et interprétation des résultats
├── pyproject.toml                    # Package Python et dépendances
├── poetry.lock                       # Versions verrouillées des dépendances
├── Makefile                          # Raccourcis pour installer, lancer et vérifier
├── .python-version                   # Version Python du projet
├── .gitignore                        # Fichiers locaux exclus du dépôt Git
├── .env.example                      # Variables d’environnement disponibles
├── .streamlit/
│   └── config.toml                   # Configuration de l’application locale
├── configs/
│   ├── retrieval.yaml                # Recherche, modèle et seuils d’abstention
│   ├── evaluation.yaml               # Valeurs de K et grilles de seuils
│   ├── logging.yaml                  # Logs techniques et affichage console
│   └── monitoring.yaml               # MLflow et journalisation des conversations
├── data/
│   └── raw/
│       ├── knowledge_base.json       # Documents de référence
│       └── eval_set.json             # Questions et sources attendues
├── src/
│   └── contextiq/
│       ├── ingestion/
│       │   ├── loader.py             # Chargement et validation des données
│       │   └── preprocessing.py      # Prétraitement lexical
│       ├── retrieval/
│       │   ├── base.py               # Interface commune des retrievers
│       │   ├── tfidf.py              # Recherche TF-IDF
│       │   ├── bm25.py               # Recherche BM25
│       │   └── embeddings.py         # Recherche sémantique locale
│       ├── generation/
│       │   ├── prompt_builder.py     # Construction du contexte et du prompt
│       │   └── extractive_answer.py  # Extrait cité ou refus
│       ├── evaluation/
│       │   ├── retrieval_metrics.py  # Precision, Recall, Hit Rate et MRR
│       │   ├── abstention_metrics.py # Évaluation des décisions de refus
│       │   ├── answer_metrics.py     # Fidélité, citations et succès des réponses
│       │   └── evaluator.py          # Évaluation des classements
│       ├── safety/
│       │   ├── abstention.py         # Politique de refus selon le score
│       │   └── prompt_injection.py   # Détection d’instructions suspectes
│       ├── monitoring/
│       │   ├── mlflow_tracking.py    # Suivi des évaluations dans MLflow
│       │   └── conversations.py      # Journal des échanges et revues humaines
│       ├── ui/
│       │   ├── app.py                # Pages de l’application Streamlit
│       │   ├── service.py            # Appels aux classes métier depuis l’interface
│       │   └── theme.py              # Présentation de l’interface
│       ├── schemas/
│       │   └── models.py             # Modèles de données Pydantic
│       ├── settings.py               # Chargement des configurations YAML
│       ├── logging.py                # Journalisation Loguru
│       └── cli.py                    # Aide et version du package
├── note/
│   └── README-0.md                    # Énoncé de l’exercice
├── notebooks/                        # Huit notebooks, décrits plus bas
├── tests/                            # Tests des composants et de la plateforme
├── reports/
│   ├── evaluation_report.md          # Résultats et analyse des erreurs
│   ├── evaluation_protocol.md        # Protocole de validation proposé
│   ├── production_monitoring.md      # Plan de suivi et traitement des retours
│   ├── project_audit.md              # Audit initial du projet
│   ├── requirements_review.md        # Correspondance avec les exigences
│   └── metrics/                      # Instantanés JSON des résultats mesurés
├── logs/                             # Journaux générés localement
└── .github/
    └── workflows/
        └── ci.yml                    # Vérifications GitHub Actions
```

La logique réutilisable se trouve dans **`src/contextiq/`**. Les notebooks et l’application appellent ces classes : les calculs métier ne sont pas dupliqués dans chaque interface. Les fichiers de `logs/` et le stockage `mlflow/` sont créés à l’utilisation et ne sont pas versionnés.

## Installer le projet

Prérequis : **Python 3.15.2**, **Poetry 2.1.3** et **Make**, disponibles sur votre machine.

Ouvrez un terminal à la racine du projet, dans le dossier contenant le `Makefile`, puis lancez :

```sh
make install
```

Ce raccourci sélectionne Python 3.15, crée ou utilise l’environnement du projet, puis installe les dépendances nécessaires à l’application et aux notebooks.

Les deux fichiers `knowledge_base.json` et `eval_set.json` doivent être présents dans `data/raw/`. Ils sont inclus dans l’archive de remise et exclus du dépôt Git : après un clonage, ajoutez-les localement avant d’utiliser le projet. Aucune clé d’API n’est nécessaire.

## Utiliser l’application

```sh
make app
```

Ouvrez **[http://127.0.0.1:8501](http://127.0.0.1:8501)**. Gardez le terminal ouvert pendant l’utilisation ; **Ctrl+C** arrête l’application.

Dans **Simulation**, commencez avec **TF-IDF** ou **BM25**, qui ne nécessitent pas de poids de modèle. La section **Des questions pour commencer** propose quatre exemples. Envoyez une question, puis ouvrez **Comprendre ce résultat** pour inspecter les scores, les sources et la décision du système.

| Page | Ce qu’elle permet |
| --- | --- |
| **Simulation** | Poser une question et lire un extrait cité ou un refus. |
| **Base documentaire** | Consulter les documents et les annotations d’évaluation. |
| **Évaluations** | Comparer Precision@K, Recall@K, Hit Rate@K et MRR@K, puis examiner les réponses et les refus. |
| **Suivi des conversations** | Relire les échanges enregistrés et ajouter une appréciation humaine. |
| **Comprendre le projet** | Lire le fonctionnement, le prompt et les rapports depuis la plateforme. |

Chaque question est traitée indépendamment ; l’historique n’est pas un contexte conversationnel. Pour comparer les trois méthodes dans **Évaluations**, préparez d’abord les embeddings. Vous pouvez sinon désélectionner cette méthode et évaluer uniquement TF-IDF et BM25.

## Parcourir les notebooks

Après l’installation, préparez un noyau identifiable dans votre éditeur :

```sh
make kernel
```

Ce raccourci enregistre l’environnement du projet sous le nom **ContextIQ**. Dans VS Code, avec les extensions Python et Jupyter, ouvrez un notebook et choisissez **ContextIQ** dans **Sélectionner un noyau / Select Kernel**. Exécutez ensuite les cellules dans l’ordre, avec **Tout exécuter / Run All** ou **Shift+Entrée**.

Chaque notebook initialise ses propres objets. L’ordre suivant permet de suivre la démarche complète :

| Notebook | Rôle |
| --- | --- |
| [01 — Exploration des données](notebooks/01_data_understanding.ipynb) | Comprendre les documents, les questions et la couverture des annotations. |
| [02 — Méthodes de recherche](notebooks/02_retrieval_experiments.ipynb) | Examiner le prétraitement, TF-IDF, BM25 et les embeddings. |
| [03 — Évaluation du retrieval](notebooks/03_retrieval_evaluation.ipynb) | Comparer les métriques à plusieurs K et analyser les classements. |
| [04 — Abstention](notebooks/04_abstention_analysis.ipynb) | Étudier les seuils, les faux refus et les acceptations hors périmètre. |
| [05 — Prompt et sécurité](notebooks/05_prompt_and_safety.ipynb) | Construire le contexte et tester les protections contre les injections. |
| [06 — Réponses extractives](notebooks/06_extractive_answers.ipynb) | Produire un extrait documentaire cité ou un refus. |
| [07 — Évaluation des réponses](notebooks/07_answer_evaluation.ipynb) | Analyser les réponses de l’exercice et mesurer la qualité du pipeline. |
| [08 — Suivi MLflow](notebooks/08_mlflow_monitoring.ipynb) | Enregistrer et comparer les évaluations locales. |

Le notebook **01** fonctionne sans poids d’embeddings. Les notebooks **02 à 08** initialisent aussi cette méthode : préparez le modèle avant de les exécuter.

### Préparer les embeddings au premier usage

Le modèle utilisé est `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

1. Dans [configs/retrieval.yaml](configs/retrieval.yaml), passez temporairement `embeddings.local_files_only` à `false`.
2. Posez une question avec **Embeddings** dans l’application, ou exécutez le notebook 02 jusqu’à son initialisation. Le premier chargement télécharge les poids et nécessite Internet.
3. Après un chargement réussi, remettez la valeur à `true` pour utiliser les poids en cache hors ligne.

Les poids sont conservés sur disque dans le cache Hugging Face, ou dans le dossier choisi avec `cache_dir`. Les vecteurs documentaires restent en mémoire pendant la session.

## Retrouver les résultats et les rapports

La page **Évaluations** présente les scores calculés sur le jeu fourni. Pour consulter l’historique des exécutions MLflow, ouvrez un second terminal à la racine du projet :

```sh
make mlflow-ui
```

Rendez-vous sur **[http://127.0.0.1:5000](http://127.0.0.1:5000)** et sélectionnez `contextiq-rag-assistant`. Le port se modifie dans `configs/monitoring.yaml` s’il est déjà utilisé. L’interface MLflow n’a pas besoin d’être ouverte pour enregistrer une évaluation.

| Document | Contenu |
| --- | --- |
| [README-Report-RAG.md](README-Report-RAG.md) | Présentation du travail, justification des choix et décisions guidées par les résultats. |
| [Rapport d’évaluation](reports/evaluation_report.md) | Mesures, erreurs observées et limites. |
| [Protocole d’évaluation](reports/evaluation_protocol.md) | Démarche proposée pour une validation indépendante. |
| [Suivi en production](reports/production_monitoring.md) | Indicateurs, revues humaines et actions après modification documentaire. |
| [Vérification du parcours utilisateur](reports/delivery_verification.md) | Installation, application et huit notebooks testés avant préparation de l’archive. |
| [Métriques vérifiées](reports/metrics/report_validation_2026-11-16.json) | Résultats détaillés du recalcul utilisé pour le rapport de présentation. |

Les résultats sont exploratoires : six documents, six questions et des seuils non calibrés indépendamment. Une citation valide ne garantit pas la pertinence de la source. Le filtre d’injection reste heuristique.

## Adapter la configuration

| Fichier | Paramètres |
| --- | --- |
| [retrieval.yaml](configs/retrieval.yaml) | K, BM25, modèle d’embeddings et seuils d’abstention. |
| [evaluation.yaml](configs/evaluation.yaml) | Valeurs de K et grilles de seuils à comparer. |
| [logging.yaml](configs/logging.yaml) | Dossier des logs, niveau, rotation, rétention et affichage console. |
| [monitoring.yaml](configs/monitoring.yaml) | Expérience MLflow, stockage, port et journal des conversations. |

Les paramètres sont chargés avec `Settings.load()` et validés par Pydantic. Pour afficher les traces techniques dans un notebook, passez `console` à `true` dans `logging.yaml` ; la valeur par défaut est `false`.

Les données et configurations peuvent être placées ailleurs avec `CONTEXTIQ_DATA_DIR` et `CONTEXTIQ_CONFIG_DIR`. Les options sont décrites dans [.env.example](.env.example). Le fichier `.env` n’est pas chargé automatiquement.

### Où sont enregistrés les échanges ?

- **`logs/contextiq-rag-assistant.log`** : traces techniques Loguru, durées et identifiants de corrélation.
- **`logs/conversations.db`** : questions, réponses, sources, paramètres et revues humaines, consultables depuis **Suivi des conversations**.
- **`mlflow/`** : paramètres, métriques agrégées et artefacts des évaluations.

Effacer l’historique visible de simulation ne supprime pas le journal persistant. La collecte se règle avec `log_conversations` dans `monitoring.yaml`. Les textes des échanges ne sont pas enregistrés dans MLflow ; leur bilan agrégé peut l’être depuis la plateforme.

## Raccourcis pour contribuer au projet

Ces vérifications sont distinctes de l’installation et du lancement :

| Commande | Action |
| --- | --- |
| `make test` | Exécuter les tests automatisés. |
| `make lint` | Vérifier la qualité du code avec Ruff. |
| `make format-check` | Vérifier le formatage sans modifier les fichiers. |
| `make format` | Appliquer le formatage. |
| `make ci` | Valider Poetry, vérifier le code et le formatage, puis exécuter les tests. |

Le workflow [GitHub Actions](.github/workflows/ci.yml) reprend les vérifications, construit le package et vérifie sa commande installée. Il ne déploie pas l’application.
